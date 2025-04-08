import random
import torch
import numpy as np
from typing import List, Optional, Tuple
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

from _types import RadarSample
from utils import matrix_to_image


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


def rotate_nearest(img, angle):
    return TF.rotate(img, angle, interpolation=InterpolationMode.NEAREST, fill=0, expand=True)

def rotate_linear(img, angle):
    return TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR, fill=0, expand=True)

def rotate_db(img, angle):
    lin_energy = 10.0 ** (img / 10.0)
    lin_rs = TF.rotate(lin_energy, angle, interpolation=InterpolationMode.BILINEAR, fill=0, expand=True)
    img_rs = torch.zeros_like(lin_rs)
    valid_mask = lin_rs > 0
    img_rs[valid_mask] = 10.0 * torch.log10(lin_rs[valid_mask])
    return img_rs


def calculate_distance(x_ant, y_ant, H, W, pixel_size):
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, dtype=torch.float32, device=torch.device('cpu')),
        torch.arange(W, dtype=torch.float32, device=torch.device('cpu')),
        indexing='ij'
    )
    return torch.sqrt((x_grid - x_ant)**2 + (y_grid - y_ant)**2) * pixel_size



def normalize_size(sample: RadarSample, target_size) -> RadarSample:
    if sample.x_ant < 0 or sample.x_ant >= sample.W or sample.y_ant < 0 or sample.y_ant >= sample.H:
        # print(f"Warning: antenna coords out of range. (x_ant={sample.x_ant}, y_ant={sample.y_ant}), (W={sample.W}, H={sample.H}) -> clamping to valid range.")
        sample.x_ant = max(0, min(sample.x_ant, sample.W - 1))
        sample.y_ant = max(0, min(sample.y_ant, sample.H - 1))
    C, H, W = sample.input_img.shape
    scale_factor = min(target_size / H, target_size / W)
    new_h, new_w = int(H * scale_factor), int(W * scale_factor)
    new_size = (new_h, new_w)
    
    reflectance = sample.input_img[0:1]  # First channel with dimension [1, H, W]
    transmittance = sample.input_img[1:2]  # Second channel with dimension [1, H, W]
    
    reflectance_resized = resize_nearest(reflectance, new_size)
    transmittance_resized = resize_nearest(transmittance, new_size)
    mask_resized = resize_nearest(sample.mask.unsqueeze(0), new_size).squeeze(0)
    
    sample.x_ant = int(sample.x_ant * scale_factor)
    sample.y_ant = int(sample.y_ant * scale_factor)
    
    sample.pixel_size /= scale_factor  # Update pixel size (divide by scale factor)
    
    sample.input_img = torch.zeros((C, target_size, target_size), dtype=torch.float32, device=torch.device('cpu'))
    sample.input_img[0:1, :new_h, :new_w] = reflectance_resized
    sample.input_img[1:2, :new_h, :new_w] = transmittance_resized
    
    if sample.output_img is not None:
        resized_output = resize_db(sample.output_img.unsqueeze(0), new_size).squeeze(0)
        padded_output = torch.zeros((target_size, target_size), dtype=torch.float32, device=torch.device('cpu'))
        padded_output[:new_h, :new_w] = resized_output
        sample.output_img = padded_output
    
    sample.H = sample.W = target_size

    sample.mask = torch.zeros((target_size, target_size), dtype=torch.float32, device=torch.device('cpu'))
    sample.mask[:new_h, :new_w] = mask_resized
    
    sample.input_img[2, :, :] = calculate_distance(sample.x_ant, sample.y_ant, sample.H, sample.W, sample.pixel_size)

    return sample



class BaseAugmentation:
    """Base class for all augmentations"""
    def __call__(self, sample: RadarSample) -> RadarSample:
        raise NotImplementedError
    
class CompositeAugmentation:
    """Subclass for all augmentations composite augmentations which mix the sample with other samples."""
    def __call__(self, sample: RadarSample, all_samples: List[RadarSample]) -> RadarSample:
        raise NotImplementedError


class GeometricAugmentation(BaseAugmentation):
    def __init__(
            self,
            flip_vertical: bool = False,
            flip_horizontal: bool = False,
            angle_range: Optional[Tuple[float, float]] = None,
            cardinal_rotation: bool = False,
            scale_range: Optional[Tuple[float, float]] = None,
    ):
        self.scale_range = scale_range
        self.angle_range = angle_range
        self.cardinal_rotation = cardinal_rotation
        self.flip_horizontal = flip_horizontal
        self.flip_vertical = flip_vertical
    
    def _apply_distance_scaling(self, sample: RadarSample, scale_factor: float) -> RadarSample:
        distances = sample.input_img[2]

        scaled_distances = distances * scale_factor
        fspl_adjustment = 20.0 * np.log10(scale_factor)

        sample.input_img[2] = scaled_distances
        sample.output_img += fspl_adjustment
        sample.pixel_size *= scale_factor
        
        return sample
    
    def _apply_rotation(self, sample: RadarSample, angle: float) -> RadarSample:
        old_H, old_W = sample.H, sample.W
        antenna_img = torch.zeros((old_H, old_W), dtype=torch.float32)
        antenna_img[int(round(sample.y_ant)), int(round(sample.x_ant))] = 100.0
        antenna_rot = rotate_linear(antenna_img.unsqueeze(0), angle).squeeze(0)
        coords = (antenna_rot > 0).nonzero(as_tuple=False)

        new_ay, new_ax = coords[antenna_rot[coords[:,0], coords[:,1]] == antenna_rot.max()][0].tolist()
        
        sample.x_ant, sample.y_ant = float(new_ax), float(new_ay)
        sample.azimuth = (sample.azimuth + angle) % 360

        reflectance = sample.input_img[0:1]  # (1, H, W)
        transmittance = sample.input_img[1:2]  # (1, H, W)

        if sample.output_img is not None:
            out_expanded = sample.output_img.unsqueeze(0)  # (1,H,W)
            rot_output = rotate_db(out_expanded, angle).squeeze(0)
            sample.output_img = rot_output

        if sample.mask is not None:
            mask_expanded = sample.mask.unsqueeze(0)
            rot_mask = rotate_nearest(mask_expanded, angle).squeeze(0)
            sample.mask = rot_mask

        rot_reflectance = rotate_nearest(reflectance, angle)
        rot_transmittance = rotate_nearest(transmittance, angle)

        _, sample.H, sample.W = rot_reflectance.shape
        rot_distance = calculate_distance(sample.x_ant, sample.y_ant, sample.H, sample.W, sample.pixel_size)

        sample.input_img = torch.cat([rot_reflectance, rot_transmittance, rot_distance.unsqueeze(0)], dim=0)
        sample = normalize_size(sample=sample, target_size=old_H)
        
        return sample


    def _apply_flipping(self, sample: RadarSample, flip_h: bool, flip_v: bool) -> RadarSample:
        if not (flip_h or flip_v):
            return sample
        
        if flip_h:
            sample.input_img = TF.hflip(sample.input_img)
        if flip_v:
            sample.input_img = TF.vflip(sample.input_img)
            
        if sample.output_img is not None:
            output_expanded = sample.output_img.unsqueeze(0)
            if flip_h:
                output_expanded = TF.hflip(output_expanded)
            if flip_v:
                output_expanded = TF.vflip(output_expanded)
            sample.output_img = output_expanded.squeeze(0)
        
        if flip_h:
            sample.x_ant = sample.W - sample.x_ant
            sample.azimuth = (180 - sample.azimuth) % 360
        if flip_v:
            sample.y_ant = sample.H - sample.y_ant
            sample.azimuth = (360 - sample.azimuth) % 360
            
        return sample
    
    def _apply_cardinal_rotation(self, sample: RadarSample) -> RadarSample:
        """
        Rotate by one of {90, 180, 270} degrees *losslessly* using torch.rot90.
        We also must update x_ant, y_ant, azimuth accordingly.
        """
        # Randomly choose 90°, 180°, or 270° (k=1,2,3). If you want to allow 0°, add k=0.
        k = random.choice([1, 2, 3])

        old_H, old_W = sample.H, sample.W
        sample.input_img = torch.rot90(sample.input_img, k, (1, 2))
        new_H, new_W = sample.input_img.shape[1], sample.input_img.shape[2]

        if k == 1: # 90 deg counter-clockwise
            new_x = sample.y_ant
            new_y = old_W - sample.x_ant - 1
            sample.azimuth = (sample.azimuth + 90) % 360
        elif k == 2: # 180 deg
            new_x = old_W - sample.x_ant - 1
            new_y = old_H - sample.y_ant - 1
            sample.azimuth = (sample.azimuth + 180) % 360
        elif k == 3:  # 270 deg
            new_x = old_H - sample.y_ant - 1
            new_y = sample.x_ant
            sample.azimuth = (sample.azimuth + 270) % 360

        sample.x_ant, sample.y_ant = new_x, new_y
        if sample.output_img is not None:
            sample.output_img = torch.rot90(sample.output_img, k, (0, 1))
        if sample.mask is not None:
            sample.mask = torch.rot90(sample.mask, k, (0, 1))

        sample.H, sample.W = new_H, new_W
        return sample

    def __call__(self, sample: RadarSample) -> RadarSample:
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


class CompositeAntennaAugmentation(CompositeAugmentation):
    def __init__(self, multi_antenna=True):
        self.multi_antenna = multi_antenna

    def __call__(self, sample: RadarSample, all_samples: List[RadarSample]) -> RadarSample:
        if self.multi_antenna:
            sample = self._apply_multi_antenna(sample, all_samples)

    def _apply_multi_antenna(self, sample: RadarSample, all_samples: List[RadarSample]) -> RadarSample:
        random_inds = np.arange(all_samples)
        np.random.shuffle(random_inds)
        
        pair = None
        for ind in random_inds:
            pair = all_samples[ind]
            if pair.ids[:3] == sample.ids[:3]:
                break
        if pair is None:
            raise ValueError(f"Sample with ids {sample.ids} has no pairs!")
        
        sample.x_ant = [sample.x_ant, pair.x_ant]
        sample.y_ant = [sample.y_ant, pair.y_ant]
        sample.azimuth = [sample.azimuth, pair.azimuth]
        
        return




class AugmentationPipeline:
    def __init__(self, augmentations: List[BaseAugmentation], p=None, training: bool = True):
        self.training = training
        self.augmentations = augmentations
        self.p = p or [1.0 for _ in augmentations]

        
    def __call__(self, sample: RadarSample, all_samples: List[RadarSample]) -> RadarSample:
        if not self.training:
            return sample
            
        for aug, aug_p in zip(self.augmentations, self.p):
            apply_aug = random.random() < aug_p
            if not apply_aug:
                continue
            if isinstance(aug, CompositeAugmentation):
                sample = aug(sample, all_samples)
            elif isinstance(aug, BaseAugmentation):
                sample = aug(sample)
        return sample


