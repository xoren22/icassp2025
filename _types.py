import os
import torch
from typing import Union, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class RadarSample:
    H: int
    W: int
    x_ant: float
    y_ant: float
    azimuth: float
    freq_MHz: float
    input_img: torch.Tensor  # In format (C, H, W)
    output_img: torch.Tensor  # In format (H, W) or (1, H, W)
    radiation_pattern: torch.Tensor
    pixel_size: float = 0.25
    mask: Union[torch.Tensor, None] = None
    ids: Optional[Tuple[int, int, int, int]] = None
    
    def copy(self):
        return RadarSample(
                    self.H,
                    self.W,
                    self.x_ant,
                    self.y_ant,
                    self.azimuth,
                    self.freq_MHz,
                    self.input_img,  
                    self.output_img, 
                    self.radiation_pattern,
                    self.pixel_size,
                    self.mask,
                    self.ids,
                )


@dataclass
class RadarSampleInputs:
    freq_MHz: float
    input_file: str
    output_file: Union[str, None]
    position_file: str
    radiation_pattern_file: str
    sampling_position : int
    ids: Optional[Tuple[int, int, int, int]] = None

    def asdict(self):
        return asdict(self)
    
    def __post_init__(self):
        if self.ids and not all(isinstance(i, int) for i in self.ids):
            raise ValueError("All IDs must be integers")
        
        if not isinstance(self.freq_MHz, (int, float)):
            raise ValueError("freq_MHz must be a number")
        
        for path_attr in ['input_file', 'output_file', 'position_file', 'radiation_pattern_file']:
            path = getattr(self, path_attr)
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
    
   