import torch

from train_module import RMSELoss
from data_module import PathlossDataset
from utils import matrix_to_image, load_model


def run_iterative_inference(
    model_ckpt_path,
    b, ant, f, sp,
    input_path,
    output_path,
    positions_path,
    buildings_path,
    radiation_path,
):
    iterative_model = load_model(model_ckpt_path)
    iterative_model.eval()
    
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
        preds = iterative_model(input_tensor)
    
    y_indices, x_indices = mask.squeeze(0).nonzero(as_tuple=True)
    y_min, y_max = y_indices.min(), y_indices.max() + 1
    x_min, x_max = x_indices.min(), x_indices.max() + 1

    valid_preds = preds[0, -1, y_min:y_max, x_min:x_max]
    valid_outputs = output_tensor[y_min:y_max, x_min:x_max]

    mask = mask.unsqueeze(0) # adding
    output_tensor = output_tensor.unsqueeze(0)

    rmse = RMSELoss().forward(preds, output_tensor, mask).detach().cpu().numpy()[0]

    return valid_preds, valid_outputs, rmse


if __name__ == "__main__":
    model_path = "models/best_model.pth"
    
    # Example building/antenna/freq/position
    t   = 1
    b   = 1
    ant = 1
    f   = 1
    sp  = 31
    
    base_dir       = "/auto/home/xoren/icassp2025/"
    input_path     = base_dir + f"data/Inputs/Task_{t}_ICASSP/"
    output_path    = base_dir + f"data/Outputs/Task_{t}_ICASSP/" 
    positions_path = base_dir + "data/Positions"
    buildings_path = base_dir + "data/Building_Details"
    radiation_path = base_dir + "data/Radiation_Patterns"

    pred, true, rmse = run_iterative_inference(
        model_ckpt_path=model_path,
        b=b,
        ant=ant,
        f=f,
        sp=sp,
        input_path=input_path,
        output_path=output_path,
        positions_path=positions_path,
        buildings_path=buildings_path,
        radiation_path=radiation_path,
        num_iterations=3,
        return_all_iterations=True
    )

    name = f"B{b}_Ant{ant}_f{f}_S{sp}_Task{t}_2.png"
    save_path = f"/auto/home/xoren/icassp2025/{name}"
    matrix_to_image(true, pred, titles=["Ground Truth", f"Prediction with Per Iteration RMSE: " + ", ".join(rmse.astype(str))], save_path=save_path)
    print(f"Per Iteration RMSEs of {name} : " + ", ".join(rmse.astype(str)))