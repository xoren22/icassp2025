import os
import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader

from model import UNetModel
from logger import TrainingLogger
from train_module import train_model
from _types import RadarSampleInputs
from data_module import PathlossDataset
from augmentations import AugmentationPipeline, GeometricAugmentation

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate pathloss prediction model')

    parser.add_argument('--num_workers', type=int, default=6, help='number of workers')
    
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID to use (default: auto-seleqct)')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='data/synthetic/', help='Data directory (default if train/val dirs not provided)')
    parser.add_argument('--train_data_dir', type=str, default=None, help='Training data directory (overrides --data_dir)')
    parser.add_argument('--val_data_dir', type=str, default=None, help='Validation data directory (if not set, a split from training data is used)')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--feature', type=str, default='transmittance', choices=['transmittance','combined'], help='which approximator feature to feed the model')
    
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
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, args.model_dir)

    def make_abs_dir(path_like):
        if path_like is None:
            return None
        return path_like if os.path.isabs(path_like) else os.path.join(BASE_DIR, path_like)

    train_data_dir = make_abs_dir(args.train_data_dir) or os.path.join(BASE_DIR, args.data_dir)
    val_data_dir = make_abs_dir(args.val_data_dir) if args.val_data_dir else None

    def enumerate_samples(data_dir):
        input_path = os.path.join(data_dir, "Inputs/Task_2_ICASSP/")
        output_path = os.path.join(data_dir, "Outputs/Task_2_ICASSP/")
        positions_path = os.path.join(data_dir, "Positions/")
        radiation_patterns_path = os.path.join(data_dir, "Radiation_Patterns/")

        samples = []
        for b in range(1, 26):
            for ant in range(1, 3):
                for f in range(1, 4):
                    for sp in range(80):
                        input_file = f"B{b}_Ant{ant}_f{f}_S{sp}.png"
                        output_file = f"B{b}_Ant{ant}_f{f}_S{sp}.png"
                        if os.path.exists(os.path.join(input_path, input_file)) and \
                           os.path.exists(os.path.join(output_path, output_file)):
                            radiation_file = f"Ant{ant}_Pattern.csv"
                            position_file = f"Positions_B{b}_Ant{ant}_f{f}.csv"

                            freq_MHz = freqs_MHz[f-1]
                            input_img_path = os.path.join(input_path, input_file)
                            output_img_path = os.path.join(output_path, output_file)
                            pos_path = os.path.join(positions_path, position_file)
                            radiation_pattern_file = os.path.join(radiation_patterns_path, radiation_file)

                            samples.append(
                                RadarSampleInputs(
                                    freq_MHz=freq_MHz,
                                    input_file=input_img_path,
                                    output_file=output_img_path,
                                    position_file=pos_path,
                                    radiation_pattern_file=radiation_pattern_file,
                                    sampling_position=sp,
                                    ids=(b, ant, f, sp),
                                )
                            )
        return samples

    train_files = enumerate_samples(train_data_dir)

    logger = TrainingLogger()
    print(f"\nLogging results at {logger.log_dir}\n\n")

    split_save_path=os.path.join(logger.log_dir, "train_val_split.pkl")
    if val_data_dir:
        val_files = enumerate_samples(val_data_dir)
    else:
        from utils import split_data_task1
        train_files, val_files = split_data_task1(train_files, val_ratio=0.20, split_save_path=split_save_path)
    print(f"Train: {len(train_files)}, Validation: {len(val_files)}")
    
    augmentations = AugmentationPipeline(
        [
            GeometricAugmentation(
                p=0.5,
                angle_range=(-30, 30),
                scale_range=(1/1.5, 1.5),
                flip_vertical=True,
                flip_horizontal=True,
                cardinal_rotation=True,
            ),
        ]
    )
    
    train_dataset = PathlossDataset(inputs_list=train_files, load_output=True, training=True, augmentations=augmentations, feature_type=args.feature)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    model = UNetModel().to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)    
    
    train_model(
        model=model, 
        train_loader=train_loader, 
        val_samples=val_files, 
        optimizer=optimizer, 
        scheduler=None,
        num_epochs=args.epochs, 
        logger=logger, 
        device=device, 
        save_dir=MODEL_SAVE_PATH, # "/dev/null"
        use_sip2net=True,
    )

if __name__ == "__main__":
    main()