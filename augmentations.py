import os
import torch
import random
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode


def resize_nearest(img, new_size):
    """
    Resize image using nearest neighbor interpolation with PyTorch.
    
    Args:
        img: numpy.ndarray input image
        new_size: tuple (height, width) for the output size
        
    Returns:
        numpy.ndarray resized image
    """
    # Convert numpy array to torch tensor
    img_tensor = torch.from_numpy(img).float()
    
    # Add batch and channel dimensions if needed
    if len(img_tensor.shape) == 2:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    elif len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    
    # Resize with PyTorch
    img_rs = TF.resize(img_tensor, new_size, interpolation=InterpolationMode.NEAREST_EXACT)
    
    # Remove batch dimension
    img_rs = img_rs.squeeze(0)
    
    # Convert back to numpy array with original dtype
    return img_rs.cpu().numpy().astype(img.dtype)


def resize_linear(img, new_size):
    """
    Resize image using bilinear interpolation with PyTorch.
    
    Args:
        img: numpy.ndarray input image
        new_size: tuple (height, width) for the output size
        
    Returns:
        numpy.ndarray resized image
    """
    # Convert numpy array to torch tensor
    img_tensor = torch.from_numpy(img).float()
    
    # Add batch and channel dimensions if needed
    if len(img_tensor.shape) == 2:
        img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    elif len(img_tensor.shape) == 3:
        img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    
    # Resize with PyTorch
    img_rs = TF.resize(img_tensor, new_size, interpolation=InterpolationMode.BILINEAR)
    
    # Remove batch dimension
    img_rs = img_rs.squeeze(0)
    
    # Convert back to numpy array with original dtype
    return img_rs.cpu().numpy().astype(img.dtype)


def resize_db(img, new_size):
    """
    Resize image in decibel scale with bilinear interpolation using PyTorch.
    
    Args:
        img: numpy.ndarray input image in dB scale
        new_size: tuple (height, width) for the output size
        
    Returns:
        numpy.ndarray resized image in dB scale
    """
    # Convert from dB to linear scale
    lin_energy = 10**(img / 10.0)
    
    # Convert numpy array to torch tensor
    lin_tensor = torch.from_numpy(lin_energy).float()
    
    # Add batch and channel dimensions if needed
    if len(lin_tensor.shape) == 2:
        lin_tensor = lin_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    elif len(lin_tensor.shape) == 3:
        lin_tensor = lin_tensor.unsqueeze(0)  # Add batch dimension
    
    # Resize with PyTorch
    lin_rs = TF.resize(lin_tensor, new_size, interpolation=InterpolationMode.BILINEAR)
    
    # Remove batch dimension
    lin_rs = lin_rs.squeeze(0)
    
    # Convert back to numpy array
    lin_rs_np = lin_rs.cpu().numpy()
    
    # Convert back to dB scale, handling zeros
    img_rs = np.zeros_like(lin_rs_np)
    valid_mask = lin_rs_np > 0
    img_rs[valid_mask] = 10.0 * np.log10(lin_rs_np[valid_mask])
    
    return img_rs.astype(img.dtype)

    



class GeometricTransform:
    def __init__(self, rotate_range=None, flip=False):
        self.rotate_range = rotate_range
        self.flip = flip

    def rotate_db_image(self, db_t, angle, mask_t):
        lin = 10.0 ** (db_t / 10.0)
        lin_rot = TF.rotate(lin, angle, interpolation=InterpolationMode.BILINEAR, expand=True)
        out = torch.zeros_like(lin_rot)
        nz = lin_rot > 0
        out[nz] = 10.0 * torch.log10(lin_rot[nz])
        return out * mask_t

    def rotate_linear(self, x_t, angle, mask_t):
        x_rot = TF.rotate(x_t, angle, interpolation=InterpolationMode.BILINEAR, expand=True)
        return x_rot * mask_t

    def rotate_mask(self, m_t, angle):
        return TF.rotate(m_t, angle, interpolation=InterpolationMode.NEAREST, expand=True)

    def __call__(self, inp_np, out_np, mask_np):
        # Each of these is [C, H, W], so we convert them to tensors directly.
        trans_t = torch.from_numpy(inp_np[0]).unsqueeze(0).float()  # shape [1,H,W]
        refl_t  = torch.from_numpy(inp_np[1]).unsqueeze(0).float()  # shape [1,H,W]
        dist_t  = torch.from_numpy(inp_np[2]).unsqueeze(0).float()  # shape [1,H,W]

        out_t   = torch.from_numpy(out_np).unsqueeze(0).float()  # shape [1,H,W], a single dB channel
        mask_t  = torch.from_numpy(mask_np).unsqueeze(0).float() # shape [1,H,W], same size as out_t

        if self.rotate_range:
            angle = np.random.uniform(*self.rotate_range)
            mask_t  = self.rotate_mask(mask_t, angle)
            trans_t = self.rotate_db_image(trans_t, angle, mask_t)
            refl_t  = self.rotate_db_image(refl_t,  angle, mask_t)
            dist_t  = self.rotate_linear(dist_t,    angle, mask_t)
            out_t   = self.rotate_db_image(out_t,   angle, mask_t)

        if self.flip:
            operator = np.random.choice([TF.hflip, TF.vflip])
            trans_t = operator(trans_t)
            refl_t  = operator(refl_t)
            dist_t  = operator(dist_t)
            out_t   = operator(out_t)
            mask_t  = operator(mask_t)

        inp_t = torch.cat([trans_t, refl_t, dist_t], dim=0)
        out_t, mask_t = out_t.squeeze(0), mask_t.squeeze(0)
        return inp_t, out_t, mask_t
    

class ResizeAug:
    def __init__(self, scale_range=(0.5, 1.5)):
        self.scale_range = scale_range

    def __call__(self, inp_np, out_np, mask_np):
        # inp_np: [3, H, W] -> channels 0 & 1 are dB, channel 2 is distance (linear)
        # out_np: [1, H, W] -> dB
        # mask_np: [1, H, W] -> discrete
        trans = torch.from_numpy(inp_np[0]).unsqueeze(0).float()   # dB
        refl  = torch.from_numpy(inp_np[1]).unsqueeze(0).float()   # dB
        dist  = torch.from_numpy(inp_np[2]).unsqueeze(0).float()   # linear
        out_t = torch.from_numpy(out_np[0]).unsqueeze(0).float()   # dB
        mask  = torch.from_numpy(mask_np[0]).unsqueeze(0).float()  # discrete

        # Pick random scale factor, compute new size
        factor = random.uniform(*self.scale_range)
        H, W = trans.shape[-2], trans.shape[-1]
        new_size = (int(H * factor), int(W * factor))

        # Helper to “do it right” for dB channels
        def resize_db(db_t):
            lin = 10**(db_t / 10.0)
            lin_rs = TF.resize(lin, new_size, InterpolationMode.BILINEAR)
            out = torch.zeros_like(lin_rs)
            nz = lin_rs > 0
            out[nz] = 10.0 * torch.log10(lin_rs[nz])
            return out

        # Resize mask with nearest
        mask = TF.resize(mask, new_size, InterpolationMode.NEAREST)

        # Resize channels
        trans = resize_db(trans) * mask
        refl  = resize_db(refl)  * mask
        dist  = TF.resize(dist, new_size, InterpolationMode.BILINEAR) * mask
        out_t = resize_db(out_t) * mask

        # Reassemble input => [3, newH, newW]
        inp_t = torch.cat([trans, refl, dist], dim=0)
        # out_t => [1, newH, newW], mask => [1, newH, newW]
        return inp_t, out_t, mask




if __name__ == "__main__":
    inpath = "/Users/xoren/icassp2025/data/Inputs/Task_2_ICASSP/"
    outpath = "/Users/xoren/icassp2025/data/Outputs/Task_2_ICASSP/"

    def show(in0, out0, mask0, nin, nout, nmask):
        fig, axes = plt.subplots(nrows=2, ncols=5)#, figsize=(15, 15))

        axes[0,0].imshow(in0[0], cmap="coolwarm")
        axes[0,1].imshow(in0[1], cmap="coolwarm")
        axes[0,2].imshow(in0[2], cmap="coolwarm")
        axes[0,3].imshow(out0, cmap="coolwarm")
        axes[0,4].imshow(mask0, cmap="coolwarm")

        axes[1,0].imshow(nin[0], cmap="coolwarm")
        axes[1,1].imshow(nin[1], cmap="coolwarm")
        axes[1,2].imshow(nin[2], cmap="coolwarm")
        axes[1,3].imshow(nout, cmap="coolwarm")
        axes[1,4].imshow(nmask, cmap="coolwarm")

        plt.tight_layout()
        plt.show()

    def readf(ids=None, fname=None):
        if ids is None and fname is None:
            raise ValueError("aa")
        fname = fname # f"B{b}_Ant{ant}_f{f}_S{sp}.png"
        ins = imread(os.path.join(inpath, fname))
        outs = imread(os.path.join(outpath, fname))
        return ins, outs

    in0, out0 = readf(fname="B1_Ant1_f1_S0.png")
    in1, out1 = readf(fname="B1_Ant1_f1_S33.png")

    trans = GeometricTransform(
        rotate_range=(15, 15),
        flip=True,
    )

    in0 = in0.transpose(2,0,1)
    mask_np = np.ones(out0.shape)
    nin, nout, nmask = trans(in0, out0, mask_np)

    show(in0, out0, mask_np, nin, nout, nmask)