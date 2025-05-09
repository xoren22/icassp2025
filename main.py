import os
import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader

from model import UNetModel
from logger import TrainingLogger
from utils import split_data_task1
from train_module import train_model
from _types import RadarSampleInputs
from data_module import PathlossDataset
from augmentations import AugmentationPipeline, GeometricAugmentation

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate pathloss prediction model')

    parser.add_argument('--num_workers', type=int, default=0, help='number of workers')
    
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID to use (default: auto-seleqct)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='data/train/', help='Data directory')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Set device
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Define paths
    freqs_MHz = [868, 1800, 3500]
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, args.data_dir)
    INPUT_PATH = os.path.join(DATA_DIR, f"Inputs/Task_1_ICASSP/")
    OUTPUT_PATH = os.path.join(DATA_DIR, f"Outputs/Task_1_ICASSP/")
    POSITIONS_PATH = os.path.join(DATA_DIR, "Positions/")
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, args.model_dir)
    
    inputs_list = []
    for b in range(1, 26):  # 25 buildings
        for ant in range(1, 3):  # 2 antenna types
            for f in range(1, 2):  # 3 frequencies
                for sp in range(80):  # 80 sampling positions
                    # Check if file exists
                    input_file = f"B{b}_Ant{ant}_f{f}_S{sp}.png"
                    output_file = f"B{b}_Ant{ant}_f{f}_S{sp}.png"
                    
                    if os.path.exists(os.path.join(INPUT_PATH, input_file)) and \
                       os.path.exists(os.path.join(OUTPUT_PATH, output_file)):
                        input_file = f"B{b}_Ant{ant}_f{f}_S{sp}.png"
                        output_file = f"B{b}_Ant{ant}_f{f}_S{sp}.png"
                        position_file = f"Positions_B{b}_Ant{ant}_f{f}.csv"

                        freq_MHz = freqs_MHz[f-1]
                        input_img_path = os.path.join(INPUT_PATH, input_file)
                        output_img_path = os.path.join(OUTPUT_PATH, output_file)
                        positions_path = os.path.join(POSITIONS_PATH, position_file)
                        
                        radar_sample_inputs = RadarSampleInputs(
                            freq_MHz=freq_MHz,
                            input_file=input_img_path,
                            output_file=output_img_path,
                            position_file=positions_path,
                            sampling_position=sp,
                            ids=(b, ant, f, sp),
                        )
                        inputs_list.append(radar_sample_inputs)




    logger = TrainingLogger()
    print(f"\nLogging results at {logger.log_dir}\n\n")

    split_save_path=os.path.join(logger.log_dir, "train_val_split.pkl")
    train_files, val_files = split_data_task1(inputs_list, split_save_path=split_save_path)
    print(f"Train: {len(train_files)}, Validation: {len(val_files)}")
    
    augmentations = AugmentationPipeline(
        [
            GeometricAugmentation(
                p=0.5,
                # angle_range=(-30, 30),
                # scale_range=(1/1.5, 1.5),
                flip_vertical=True,
                flip_horizontal=True,
                cardinal_rotation=True,
            ),
        ]
    )
    
    train_dataset = PathlossDataset(inputs_list=train_files, augmentations=augmentations)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    model = UNetModel().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=3e-5)    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500, verbose=True)
    
    train_model(
        model=model, 
        train_loader=train_loader, 
        val_samples=val_files, 
        optimizer=optimizer, 
        scheduler=scheduler,
        num_epochs=args.epochs, 
        logger=logger, 
        device=device, 
        save_dir=MODEL_SAVE_PATH, # "/dev/null"
        use_sip2net=False,
    )

if __name__ == "__main__":
    main()