import os
import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader

from model import UNetModel
from logger import TrainingLogger
from utils import split_data_task2
from train_module import train_model
from data_module import PathlossDataset


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate pathloss prediction model')

    parser.add_argument('--gpu', type=int, default=None, help='GPU ID to use (default: auto-select)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    
    args = parser.parse_args()
    
    # Set device
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Clear CUDA cache at the start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Define paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, args.data_dir)
    INPUT_PATH = os.path.join(DATA_DIR, f"Inputs/Task_2_ICASSP/")
    OUTPUT_PATH = os.path.join(DATA_DIR, f"Outputs/Task_2_ICASSP/")
    POSITIONS_PATH = os.path.join(DATA_DIR, "Positions/")
    BUILDING_DETAILS_PATH = os.path.join(DATA_DIR, "Building_Details/")
    RADIATION_PATTERNS_PATH = os.path.join(DATA_DIR, "Radiation_Patterns/")
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, args.model_dir)
    
    file_list = []
    for b in range(1, 26):  # 25 buildings
        for ant in range(1, 3):  # 2 antenna types
            for f in range(1, 4):  # 3 frequencies
                for sp in range(80):  # 80 sampling positions
                    # Check if file exists
                    input_file = f"B{b}_Ant{ant}_f{f}_S{sp}.png"
                    output_file = f"B{b}_Ant{ant}_f{f}_S{sp}.png"
                    
                    if os.path.exists(os.path.join(INPUT_PATH, input_file)) and \
                       os.path.exists(os.path.join(OUTPUT_PATH, output_file)):
                        file_list.append((b, ant, f, sp))

    logger = TrainingLogger()
    print(f"\nLogging results at {logger.log_dir}\n\n")

    split_save_path=os.path.join(logger.log_dir, "train_val_split.pkl")
    train_files, val_files = split_data_task2(file_list, val_freqs=3, split_save_path=split_save_path)
    print(f"Train: {len(train_files)}, Validation: {len(val_files)}")
    
    train_dataset = PathlossDataset(
        file_list=train_files, 
        input_path=INPUT_PATH, 
        output_path=OUTPUT_PATH, 
        positions_path=POSITIONS_PATH, 
        buildings_path=BUILDING_DETAILS_PATH, 
        radiation_path=RADIATION_PATTERNS_PATH, 
        load_output=True, training=True
    )
    
    val_dataset = PathlossDataset(
        file_list=val_files, 
        input_path=INPUT_PATH, 
        output_path=OUTPUT_PATH, 
        positions_path=POSITIONS_PATH, 
        buildings_path=BUILDING_DETAILS_PATH, 
        radiation_path=RADIATION_PATTERNS_PATH, 
        load_output=True, training=False
    )
    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    model = UNetModel().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)
    
    train_model(
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        optimizer=optimizer, 
        scheduler=scheduler,
        num_epochs=args.epochs, 
        logger=logger, 
        device=device, 
        save_dir=MODEL_SAVE_PATH,
    )
   
if __name__ == "__main__":
    main()