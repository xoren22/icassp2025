import torch
import pickle as pkl

from data_module import PathlossDataset
from utils import matrix_to_image, load_model


def run_inference(
    model_ckpt_path,
    b, ant, f, sp,
    input_path,
    output_path,
    positions_path,
    buildings_path,
    radiation_path,
):
    model = load_model(model_ckpt_path)
    model.eval()
    
    file_list = [(b, ant, f, sp)]
    dataset = PathlossDataset(
        training=False,
        load_output=True,
        file_list=file_list,
        input_path=input_path,
        output_path=output_path,       # or None if truly unused
        positions_path=positions_path,
        buildings_path=buildings_path,
        radiation_path=radiation_path
    )
    
    input_tensor, output_tensor, mask = dataset[0]
    input_tensor = input_tensor.unsqueeze(0)
    with torch.no_grad():
        preds = model(input_tensor)
    
    y_indices, x_indices = mask.squeeze(0).nonzero(as_tuple=True)
    y_min, y_max = y_indices.min(), y_indices.max() + 1
    x_min, x_max = x_indices.min(), x_indices.max() + 1

    valid_preds = preds[0, y_min:y_max, x_min:x_max]
    valid_outputs = output_tensor[y_min:y_max, x_min:x_max]

    return valid_preds, valid_outputs


if __name__ == "__main__":
    model_path = '/auto/home/xoren/icassp2025/models/best_model.pth'
    split_path = "logs/2025-03-13_20-08-26/train_val_split.pkl"
    with open(split_path, "rb") as f:
        split = pkl.load(f)
        val_files = split['val_files']
    
    # Example building/antenna/freq/position
    t   = 2
    idx = 0
    b, ant, f, sp = val_files[idx]
    
    base_dir       = "/auto/home/xoren/icassp2025/"
    input_path     = base_dir + f"data/Inputs/Task_{t}_ICASSP/"
    output_path    = base_dir + f"data/Outputs/Task_{t}_ICASSP/" 
    positions_path = base_dir + "data/Positions"
    buildings_path = base_dir + "data/Building_Details"
    radiation_path = base_dir + "data/Radiation_Patterns"

    pred, true = run_inference(
        b=b,
        ant=ant,
        f=f,
        sp=sp,
        input_path=input_path,
        output_path=output_path,
        model_ckpt_path=model_path,
        positions_path=positions_path,
        buildings_path=buildings_path,
        radiation_path=radiation_path,
    )

    name = f"B{b}_Ant{ant}_f{f}_S{sp}_Task{t}_2.png"
    save_path = f"/auto/home/xoren/icassp2025/foo/{name}"
    matrix_to_image(true, pred, titles=["Ground Truth", f"Prediction"], save_path=save_path)