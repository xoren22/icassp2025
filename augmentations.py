import random
import torch
import numpy as np
from typing import List, Optional, Tuple
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

from _types import RadarSample
from featurizer import calculate_fspl

def resize_nearest(img, new_size):
    return TF.resize(img, new_size, interpolation=InterpolationMode.NEAREST_EXACT)


def resize_linear(img, new_size):
    return TF.resize(img, new_size, interpolation=InterpolationMode.BILINEAR)


def resize_db(img, new_size):
    lin_energy = 10.0 ** (img / 10.0)
    lin_rs = TF.resize(lin_energy, new_size, interpolation=InterpolationMode.BILINEAR)
    img_rs = torch.zeros_like(lin_rs)
    valid_mask = lin_rs > 0
    img_rs[valid_mask] = 10.0 * torch.log10(lin_rs[valid_mask])
    
    return img_rs


class BaseAugmentation:
    """Base class for all augmentations"""
    def __call__(self, sample: RadarSample) -> RadarSample:
        raise NotImplementedError


class GeometricAugmentation(BaseAugmentation):
    def __init__(
            self, 
            p: float = 0.5,
            flip_vertical: bool = False, 
            flip_horizontal: bool = False, 
            angle_range: Optional[Tuple[float, float]] = None,
            scale_range: Optional[Tuple[float, float]] = None, 
        ):
        """
        Args:
            angle_range: Range of rotation angles in degrees, None to disable rotation
            flip_horizontal: Enable horizontal flipping
            flip_vertical: Enable vertical flipping
            scale_range: Range of scale factors, None to disable scaling
            p: Overall probability of applying at least one augmentation
        """
        self.p = p
        self.angle_range = angle_range
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.scale_range = scale_range
    
    def _apply_scaling(self, sample: RadarSample, scale_factor: float) -> RadarSample:
        distances = sample.input_img[0]
        output_img = sample.output_img

        scaled_distances = distances * scale_factor
        fspl_adjusted_output = output_img + 20.0 * np.log10(scale_factor)

        sample.input_img[0] = scaled_distances
        sample.output_img += fspl_adjusted_output
        
        return sample
    
    def _rotate_db_channel(self, channel, angle):
        lin = 10.0 ** (channel / 10.0)  # Convert dB to linear
        lin_rot = TF.rotate(lin, angle, interpolation=InterpolationMode.BILINEAR)
        # Convert back to dB
        out = torch.zeros_like(lin_rot)
        nz = lin_rot > 0
        out[nz] = 10.0 * torch.log10(lin_rot[nz])
        return out
    
    def _apply_rotation(self, input_img, output_img, x_ant, y_ant, azimuth, H, W, angle):
        """Apply rotation with a specific angle to the input and output images.
        
        Args:
            input_img: Input image tensor [C, H, W]
            output_img: Output image tensor [H, W] or None
            x_ant, y_ant: Antenna coordinates
            azimuth: Antenna azimuth angle
            H, W: Image height and width
            angle: Rotation angle in degrees
            
        Returns:
            Tuple containing rotated (input_img, output_img, x_ant, y_ant, azimuth)
        """
        # For channels 0 and 1 which are in dB scale
        rotated_inputs = []
        
        # Handle dB channels (reflectance and transmittance)
        for i in range(2):
            channel = input_img[i:i+1]  # Keep dimension [1, H, W]
            rotated_inputs.append(self._rotate_db_channel(channel, angle))
            
        # Handle linear channel (distance)
        distance = input_img[2:3]  # Keep dimension [1, H, W]
        rotated_distance = TF.rotate(distance, angle, interpolation=InterpolationMode.BILINEAR)
        rotated_inputs.append(rotated_distance)
        
        # Combine rotated inputs
        rotated_input = torch.cat(rotated_inputs, dim=0)
        
        # Rotate output image if it exists
        rotated_output = output_img
        if output_img is not None:
            output_expanded = output_img.unsqueeze(0)  # [1, H, W]
            rotated_output = self._rotate_db_channel(output_expanded, angle).squeeze(0)
        
        # Rotate the antenna coordinates
        angle_rad = np.radians(angle)
        x_center, y_center = W / 2, H / 2
        x_rel, y_rel = x_ant - x_center, y_ant - y_center
        
        # Rotate around center
        new_x_rel = x_rel * np.cos(angle_rad) - y_rel * np.sin(angle_rad)
        new_y_rel = x_rel * np.sin(angle_rad) + y_rel * np.cos(angle_rad)
        
        # Translate back
        x_ant = new_x_rel + x_center
        y_ant = new_y_rel + y_center
        
        # Update azimuth
        azimuth = (azimuth + angle) % 360
        
        return rotated_input, rotated_output, x_ant, y_ant, azimuth
    
    def _apply_flipping(self, input_img, output_img, x_ant, y_ant, azimuth, H, W, flip_h, flip_v):
        """Apply flipping with specific directions to the input and output images.
        
        Args:
            input_img: Input image tensor [C, H, W]
            output_img: Output image tensor [H, W] or None
            x_ant, y_ant: Antenna coordinates
            azimuth: Antenna azimuth angle
            H, W: Image height and width
            flip_h: Whether to flip horizontally
            flip_v: Whether to flip vertically
            
        Returns:
            Tuple containing flipped (input_img, output_img, x_ant, y_ant, azimuth)
        """
        # If neither flip is selected, return unchanged
        if not (flip_h or flip_v):
            return input_img, output_img, x_ant, y_ant, azimuth
        
        # Apply flips to input image
        flipped_input = input_img
        if flip_h:
            flipped_input = TF.hflip(flipped_input)
        if flip_v:
            flipped_input = TF.vflip(flipped_input)
            
        # Apply flips to output image if it exists
        flipped_output = output_img
        if output_img is not None:
            output_expanded = output_img.unsqueeze(0)
            if flip_h:
                output_expanded = TF.hflip(output_expanded)
            if flip_v:
                output_expanded = TF.vflip(output_expanded)
            flipped_output = output_expanded.squeeze(0)
        
        # Update antenna coordinates and azimuth
        if flip_h:
            x_ant = W - x_ant
            azimuth = (180 - azimuth) % 360
        if flip_v:
            y_ant = H - y_ant
            azimuth = (360 - azimuth) % 360
            
        return flipped_input, flipped_output, x_ant, y_ant, azimuth
        
    def __call__(self, sample: RadarSample) -> RadarSample:
        """Apply geometric augmentations to the sample.
        
        This method orchestrates the application of scaling, rotation, and flipping
        transformations in a consistent order. It determines which transformations
        to apply and with what parameters, then delegates the actual application
        to specialized methods.
        
        Args:
            sample: RadarSample instance
            
        Returns:
            Augmented RadarSample instance
        """
        # Decide whether to apply any augmentation
        if random.random() > self.p:
            return sample
            
        # Make a deep copy of the original sample attributes we need to modify
        input_img = sample.input_img.clone()
        output_img = sample.output_img.clone() if sample.output_img is not None else None
        x_ant, y_ant = sample.x_ant, sample.y_ant
        azimuth = sample.azimuth
        H, W = sample.H, sample.W
        
        # 1. Determine scaling parameters and apply if enabled
        if self.scale_range is not None:
            # Determine random scale factor
            scale_factor = random.uniform(*self.scale_range)
            
            # Apply scaling
            input_img, output_img, x_ant, y_ant = self._apply_scaling(
                input_img, output_img, x_ant, y_ant, H, W, scale_factor
            )
        
        # 2. Determine rotation parameters and apply if enabled
        if self.angle_range is not None:
            # Determine random angle
            angle = random.uniform(*self.angle_range)
            
            # Apply rotation
            input_img, output_img, x_ant, y_ant, azimuth = self._apply_rotation(
                input_img, output_img, x_ant, y_ant, azimuth, H, W, angle
            )
        
        # 3. Determine flipping parameters and apply if enabled
        flip_h = self.flip_horizontal and random.random() < 0.5
        flip_v = self.flip_vertical and random.random() < 0.5
        
        if flip_h or flip_v:
            # Apply flipping
            input_img, output_img, x_ant, y_ant, azimuth = self._apply_flipping(
                input_img, output_img, x_ant, y_ant, azimuth, H, W, flip_h, flip_v
            )
        
        # Update the sample with all modifications
        sample.input_img = input_img
        sample.output_img = output_img
        sample.x_ant = x_ant
        sample.y_ant = y_ant
        sample.azimuth = azimuth
        
        return sample


class AugmentationPipeline:
    """Pipeline for applying multiple augmentations in sequence"""
    def __init__(self, augmentations: List[BaseAugmentation], training: bool = True):
        """
        Args:
            augmentations: List of augmentation instances
            training: Whether to apply augmentations (only in training mode)
        """
        self.augmentations = augmentations
        self.training = training
        
    def __call__(self, sample: RadarSample) -> RadarSample:
        """Apply all augmentations in sequence to the sample.
        
        Args:
            sample: RadarSample instance
            
        Returns:
            Augmented RadarSample instance
        """
        if not self.training:
            return sample
            
        # Apply each augmentation in sequence
        for aug in self.augmentations:
            sample = aug(sample)
            
        return sample


# Example usage in the dataset class:
"""
class PathlossDataset(Dataset):
    def __init__(self, file_list, input_path, positions_path, buildings_path, radiation_path, 
                 output_path=None, img_size=640, training=False, load_output=True,
                 augment=False):
        # ... existing initialization ...
        self.training = training
        self.augment = augment
        
        # Setup augmentation pipeline if needed
        if self.augment and self.training:
            self.augmentation = AugmentationPipeline([
                # Combined geometric augmentation
                GeometricAugmentation(
                    angle_range=(-15, 15),          # Enable rotation with ±15° range
                    flip_horizontal=True,           # Enable horizontal flipping
                    flip_vertical=True,             # Enable vertical flipping
                    scale_range=(0.8, 1.2),         # Enable scaling with 0.8-1.2x range
                    p=0.7                           # 70% probability of applying some transformation
                )
                # You can add other types of augmentations here if needed
            ], training=self.training)
        else:
            self.augmentation = None
        
    def __getitem__(self, idx):
        b, ant, f, sp = self.file_list[idx]
        sample = self.read_sample(b, ant, f, sp)
        sample, mask = self._normalize_size(sample)
        
        # Apply augmentations if enabled
        if self.augmentation is not None:
            sample = self.augmentation(sample)
        
        # Continue with feature extraction...
        # ...
"""