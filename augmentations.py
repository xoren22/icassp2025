import math
import torch
import random
import numpy as np
from typing import List, Optional, Tuple
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

from _types import RadarSample
from featurizer import calculate_transmittance_loss


def resize_nearest(img, new_size):
    return TF.resize(img.unsqueeze(0), new_size, interpolation=InterpolationMode.NEAREST_EXACT).squeeze(0)

def resize_linear(img, new_size):
    return TF.resize(img.unsqueeze(0), new_size, interpolation=InterpolationMode.BILINEAR).squeeze(0)

def resize_db(img, new_size):
    img = img.unsqueeze(0)
    img = img.to(torch.float64)
    lin_energy = 10.0 ** (-img / 10.0)
    lin_rs = TF.resize(lin_energy, new_size, interpolation=InterpolationMode.BILINEAR)
    img_rs = torch.zeros_like(lin_rs)
    valid_mask = lin_rs > 0
    img_rs[valid_mask] = -10.0 * torch.log10(lin_rs[valid_mask])

    return img_rs.squeeze(0)


def rotate_nearest(img, angle):
    return TF.rotate(img.unsqueeze(0), angle, interpolation=InterpolationMode.NEAREST, fill=0, expand=True).squeeze(0)

def rotate_linear(img, angle):
    return TF.rotate(img.unsqueeze(0), angle, interpolation=InterpolationMode.BILINEAR, fill=0, expand=True).squeeze(0)

def rotate_db(img, angle):
    img = img.to(torch.float64).unsqueeze(0)
    lin_energy = 10.0 ** (-img / 10.0)
    lin_rs = TF.rotate(lin_energy, angle, interpolation=InterpolationMode.BILINEAR, fill=0, expand=True)
    img_rs = torch.zeros_like(lin_rs)
    valid_mask = lin_rs > 0
    img_rs[valid_mask] = -10.0 * torch.log10(lin_rs[valid_mask])
    return img_rs.squeeze(0)



def normalize_size(sample: RadarSample, target_size) -> RadarSample:
    if sample.x_ant < 0 or sample.x_ant >= sample.W or sample.y_ant < 0 or sample.y_ant >= sample.H:
        b, ant, f, sp = sample.ids
        raise ValueError(f"Antenna out of the bouds for B{b}_Ant{ant}_f{f}_S{sp}.png")

    H, W = sample.reflectance.shape
    scale_factor = min(target_size / H, target_size / W)
    new_h, new_w = int(H * scale_factor), int(W * scale_factor)
    new_size = (new_h, new_w)
    
    reflectance = sample.reflectance  # First channel with dimension [1, H, W]
    transmittance = sample.transmittance  # Second channel with dimension [1, H, W]
    
    sample.x_ant = int(sample.x_ant * scale_factor)
    sample.y_ant = int(sample.y_ant * scale_factor)
    
    sample.pixel_size /= scale_factor  # Update pixel size (divide by scale factor)
    sample.reflectance = torch.zeros((target_size, target_size), dtype=torch.float32, device=torch.device('cpu'))
    sample.reflectance[:new_h, :new_w] = resize_nearest(reflectance, new_size)

    sample.transmittance = torch.zeros((target_size, target_size), dtype=torch.float32, device=torch.device('cpu'))
    sample.transmittance[:new_h, :new_w] = resize_nearest(transmittance, new_size)
    
    if sample.output_img is not None:
        resized_output = resize_db(sample.output_img, new_size)
        padded_output = torch.zeros((target_size, target_size), dtype=torch.float32, device=torch.device('cpu'))
        padded_output[:new_h, :new_w] = resized_output
        sample.output_img = padded_output
    
    sample.H = sample.W = target_size
    resized_mask = resize_nearest(sample.mask, new_size)
    sample.mask = torch.zeros((target_size, target_size), dtype=torch.float32, device=torch.device('cpu'))
    sample.mask[:new_h, :new_w] = resized_mask
    
    return sample



class BaseAugmentation:
    """Base class for all augmentations"""
    def __call__(self, sample: RadarSample) -> RadarSample:
        raise NotImplementedError


class GeometricAugmentation(BaseAugmentation):
    def __init__(
            self,
            p: float = 0.5,
            wall_p: float = 0.5,
            transmittance_range: Optional[Tuple[int, int]] = None,
            flip_vertical: bool = False,
            flip_horizontal: bool = False,
            angle_range: Optional[Tuple[float, float]] = None,
            cardinal_rotation: bool = False,
            scale_range: Optional[Tuple[float, float]] = None,

    ):
        self.p = p
        self.wall_p = wall_p
        self.scale_range = scale_range
        self.angle_range = angle_range
        self.cardinal_rotation = cardinal_rotation
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
        self.transmittance_range = transmittance_range or (2, 12)
    
    def _apply_distance_scaling(self, sample: RadarSample, scale_factor: float) -> RadarSample:
        sample.pixel_size *= scale_factor
        sample.output_img += 20.0 * math.log10(scale_factor)
        
        return sample
    
    def _apply_rotation(self, sample: RadarSample, angle: float) -> RadarSample:
        old_H, old_W = sample.H, sample.W
        antenna_img = torch.zeros((old_H, old_W), dtype=torch.float32)
        antenna_img[int(round(sample.y_ant)), int(round(sample.x_ant))] = 100.0
        antenna_rot = rotate_linear(antenna_img, angle)
        coords = (antenna_rot > 0).nonzero(as_tuple=False)

        new_ay, new_ax = coords[antenna_rot[coords[:,0], coords[:,1]] == antenna_rot.max()][0].tolist()
        
        sample.x_ant, sample.y_ant = float(new_ax), float(new_ay)
        rot_reflectance = rotate_nearest(sample.reflectance, angle)
        rot_transmittance = rotate_nearest(sample.transmittance, angle)

        if sample.output_img is not None:
            out_expanded = sample.output_img  # (1,H,W)
            rot_output = rotate_db(out_expanded, angle)
            sample.output_img = rot_output

        if sample.mask is not None:
            mask_expanded = sample.mask
            rot_mask = rotate_nearest(mask_expanded, angle)
            sample.mask = rot_mask

        new_H, new_W = rot_reflectance.shape
        sample.H, sample.W = new_H, new_W
        sample.reflectance = rot_reflectance
        sample.transmittance = rot_transmittance

        sample = normalize_size(sample=sample, target_size=old_H)

        return sample


    def _apply_flipping(self, sample: RadarSample, flip_h: bool, flip_v: bool) -> RadarSample:
        if not (flip_h or flip_v):
            return sample
        
        if flip_h:
            sample.reflectance = TF.hflip(sample.reflectance)
            sample.transmittance = TF.hflip(sample.transmittance)
        if flip_v:
            sample.reflectance = TF.vflip(sample.reflectance)
            sample.transmittance = TF.vflip(sample.transmittance)
            
        if sample.output_img is not None:
            output_expanded = sample.output_img
            if flip_h:
                output_expanded = TF.hflip(output_expanded)
            if flip_v:
                output_expanded = TF.vflip(output_expanded)
            sample.output_img = output_expanded
        
        if flip_h:
            sample.x_ant = sample.W - sample.x_ant
        if flip_v:
            sample.y_ant = sample.H - sample.y_ant
            
        return sample
    
    def _apply_cardinal_rotation(self, sample: RadarSample) -> RadarSample:
        """
        Rotate by one of {90, 180, 270} degrees *losslessly* using torch.rot90.
        We also must update x_ant, y_ant accordingly.
        """
        # Randomly choose 90°, 180°, or 270° (k=1,2,3). If you want to allow 0°, add k=0.
        k = random.choice([1, 2, 3])

        old_H, old_W = sample.H, sample.W
        sample.reflectance = torch.rot90(sample.reflectance, k, (0, 1))
        sample.transmittance = torch.rot90(sample.transmittance, k, (0, 1))
        new_H, new_W = sample.reflectance.shape[0], sample.reflectance.shape[1]

        if k == 1: # 90 deg counter-clockwise
            new_x = sample.y_ant
            new_y = old_W - sample.x_ant - 1
        elif k == 2: # 180 deg
            new_x = old_W - sample.x_ant - 1
            new_y = old_H - sample.y_ant - 1
        elif k == 3:  # 270 deg
            new_x = old_H - sample.y_ant - 1
            new_y = sample.x_ant

        sample.x_ant, sample.y_ant = new_x, new_y
        if sample.output_img is not None:
            sample.output_img = torch.rot90(sample.output_img, k, (0, 1))
        if sample.mask is not None:
            sample.mask = torch.rot90(sample.mask, k, (0, 1))

        sample.H, sample.W = new_H, new_W
        return sample

    def _apply_walls(self, sample: RadarSample, transmittance_range: Tuple[int, int]) -> RadarSample:
        H, W = sample.H, sample.W
        walls = torch.zeros((H, W))
        nv = random.randint(1, max(2, W // 3))
        nh = random.randint(1, max(2, H // 3))
        v_cols = torch.randperm(W)[:nv]
        h_rows = torch.randperm(H)[:nh]
        vmax = random.randint(*transmittance_range)
        walls[:, v_cols] = torch.randint(1, vmax, (nv,)).float()
        walls[h_rows, :] = torch.randint(1, vmax, (nh,)).float().unsqueeze(1)

        m = sample.mask == 1
        sample.transmittance[m] += walls[m]

        loss = calculate_transmittance_loss(walls, sample.x_ant, sample.y_ant)
        if sample.output_img is not None:
            sample.output_img[m] += loss[m]
        
        return sample

    def __call__(self, sample: RadarSample) -> RadarSample:
        if random.random() > self.p:
            return sample
        
        if random.random() < self.wall_p:
            sample = self._apply_walls(sample, self.transmittance_range)

        if self.scale_range is not None:
            scale_factor = random.uniform(*self.scale_range)
            sample = self._apply_distance_scaling(sample, scale_factor)

        if self.cardinal_rotation:
            sample = self._apply_cardinal_rotation(sample)

        if self.angle_range is not None:
            angle = random.uniform(*self.angle_range)
            sample = self._apply_rotation(sample, angle)

        flip_h = self.flip_horizontal and (random.random() < 0.5)
        flip_v = self.flip_vertical and (random.random() < 0.5)
        if flip_h or flip_v:
            sample = self._apply_flipping(sample, flip_h, flip_v)

        return sample


class AugmentationPipeline:
    """Pipeline for applying multiple augmentations in sequence"""
    def __init__(self, augmentations: List[BaseAugmentation], training: bool = True):
        """
        Args:
            augmentations: List of augmentation instances
            training: Whether to apply augmentations (only in training mode)
        """
        self.training = training
        self.augmentations = augmentations
        
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


