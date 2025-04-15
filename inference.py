import torch
from typing import List, Union, Dict

from utils import load_model
from _types import RadarSampleInputs
from featurizer import featurize_inputs
from augmentations import normalize_size, resize_linear
from data_module import read_sample, IMG_TARGET_SIZE, INITIAL_PIXEL_SIZE


class PathlossPredictor:
    def __init__(
        self,
        model=None,
        model_ckpt_path=None,
    ):
        if model is not None:
            self.model = model
        elif model_ckpt_path is not None:
            self.model = load_model(model_ckpt_path)
        else:
            raise ValueError("Either model or model_ckpt_path must be provided")
        self.sample_cache = {}
    
    def predict(
        self,
        input_samples: Union[Dict, RadarSampleInputs, List[Union[Dict, RadarSampleInputs]]],
        use_cache=True,
    ):  
        self.model.eval()
        input_samples = list(input_samples)

        device = next(self.model.parameters()).device

        masks = []
        old_hw_dims = []
        norm_hw_dims = []
        batched_tensors = []

        for sample_input in input_samples:
            cache_key = sample_input['input_file']
            if use_cache and cache_key in self.sample_cache:
                cached_data = self.sample_cache[cache_key]
                old_hw_dims.append(cached_data['old_hw'])
                norm_hw_dims.append(cached_data['norm_hw'])
                masks.append(cached_data['mask'].detach().clone())
                batched_tensors.append(cached_data['tensor'].detach().clone())
            else:
                sample = read_sample(sample_input)
                old_h, old_w = sample.H, sample.W

                sample = normalize_size(sample, IMG_TARGET_SIZE)
                
                mask = sample.mask
                scaling_factor = INITIAL_PIXEL_SIZE / sample.pixel_size
                norm_h, norm_w = int(old_h * scaling_factor), int(old_w * scaling_factor)

                input_tensor = featurize_inputs(sample).to(device)

                if use_cache:
                    self.sample_cache[cache_key] = {
                        'old_hw': (old_h, old_w),
                        'norm_hw': (norm_h, norm_w),
                        'mask': mask.detach().clone(),
                        'tensor': input_tensor.detach().clone(),
                    }

                batched_tensors.append(input_tensor)
                masks.append(mask)
                old_hw_dims.append((old_h, old_w))
                norm_hw_dims.append((norm_h, norm_w))
        
        batch_tensor = torch.stack(batched_tensors, dim=0)  # [B, C, H, W]
        with torch.no_grad():
            preds = self.model(batch_tensor)  

        results = []
        for i in range(len(input_samples)):
            pred_i = preds[i]  # shape ~ [C, H, W]
            mask_i = masks[i]
            (old_h, old_w) = old_hw_dims[i]
            (norm_h, norm_w) = norm_hw_dims[i]

            pred_i = pred_i.squeeze(0)  # -> [H, W]
            pred_i = pred_i[torch.where(mask_i == 1)].reshape(norm_h, norm_w)
            pred_i = resize_linear(pred_i.unsqueeze(0), new_size=(old_h, old_w)).squeeze(0)
            results.append(pred_i)

        return results



if __name__ == "__main__":
    import numpy as np
    import pickle as pkl
    from skimage.io import imread
    from utils import matrix_to_image

    model_path = '/Users/xoren/icassp2025/models/best_model.pth'
    split_file = '/Users/xoren/icassp2025/logs/2025-03-28_02-10-49/train_val_split.pkl'
    with open(split_file, "rb") as f:
        split = pkl.load(f)
    
    val_samples = [s.asdict() for s in split["val_inputs"]]
   
    val_samples = list(np.random.choice(val_samples, 3))
    val_targets = [torch.from_numpy(imread(s.pop("output_file"))).to(torch.float32) for s in val_samples]

    model = PathlossPredictor(model_ckpt_path=model_path)
    batched_pred = model.predict(val_samples)
    for s, pred, tgt in zip(val_samples, batched_pred, val_targets):
        b, ant, f, sp = s['ids']
        fname = f"B{b}_Ant{ant}_f{f}_S{sp}.png"
        save_path = f"/Users/xoren/icassp2025/foo/{fname}"
        matrix_to_image(tgt, pred, save_path=save_path)

    tgt_samples_flat = np.concatenate([A.flatten() for A in val_targets])
    pred_samples_flat = np.concatenate([A.flatten() for A in batched_pred])
    
    val_rmse = np.sqrt(np.mean(np.square(pred_samples_flat - tgt_samples_flat)))