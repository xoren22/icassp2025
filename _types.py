import os
import torch
from dataclasses import dataclass, asdict
from typing import Union, Tuple, Optional, List


@dataclass
class RadarSample:
    H: int
    W: int
    x_ant: float
    y_ant: float
    freq_MHz: float
    reflectance: torch.Tensor  # In format (H, W)
    transmittance: torch.Tensor  # In format (H, W)
    output_img: torch.Tensor  # In format (H, W)
    pixel_size: float = 0.25
    mask: Union[torch.Tensor, None] = None
    ids: Optional[List[Tuple[int, int, int, int]]] = None

    def copy(self):
        return RadarSample(
                    H=self.H,
                    W=self.W,
                    x_ant=self.x_ant,
                    y_ant=self.y_ant,
                    freq_MHz=self.freq_MHz,
                    reflectance=self.reflectance,  
                    transmittance=self.transmittance,  
                    output_img=self.output_img, 
                    pixel_size=self.pixel_size,
                    mask=self.mask,
                    ids=self.ids,
                )

@dataclass
class RadarSampleInputs:
    freq_MHz: float
    input_file: str
    output_file: Union[str, None]
    position_file: str
    sampling_position : int
    ids: Tuple[int, int, int, int]

    def asdict(self):
        return asdict(self)
    
    def __post_init__(self):
        if self.ids and not all(isinstance(i, int) for i in self.ids):
            raise ValueError("All IDs must be integers")
        
        if not isinstance(self.freq_MHz, (int, float)):
            raise ValueError("freq_MHz must be a number")
        
        for path_attr in ['input_file', 'position_file']:
            path = getattr(self, path_attr)
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path}")
  
   