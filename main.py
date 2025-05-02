# main.py
import os, argparse, torch, torch.optim as optim
from torch.utils.data import DataLoader

from model import PathLossNet 
from logger import TrainingLogger
from utils import split_data_task1
from train_module import train_model  
from _types import RadarSampleInputs
from data_module import PathlossDataset
from augmentations import AugmentationPipeline, GeometricAugmentation, RandomWallsAugmentation


def parse_args():
    p = argparse.ArgumentParser("Path-loss training")
    p.add_argument('--num_workers',   type=int,   default=6)
    p.add_argument('--gpu',           type=int,   default=None)
    p.add_argument('--batch_size',    type=int,   default=8)
    p.add_argument('--epochs',        type=int,   default=2000)
    p.add_argument('--lr',            type=float, default=3e-4)
    p.add_argument('--data_dir',      type=str,   default='data/train/')
    p.add_argument('--model_dir',     type=str,   default='models')
    # --- NEW flags --------------------------------------------------------
    p.add_argument('--use_selector',  type=int,   default=1,
                   help='1 → learn sampler (two-head); 0 → external masks')
    p.add_argument('--k_frac',        type=float, default=0.005)
    p.add_argument('--tau',           type=float, default=0.7)
    return p.parse_args()


def main():
    args = parse_args()
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    freqs_MHz = [868, 1800, 3500]
    BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR  = os.path.join(BASE_DIR, args.data_dir)
    INPUT_PATH  = os.path.join(DATA_DIR, "Inputs/Task_1_ICASSP")
    OUTPUT_PATH = os.path.join(DATA_DIR, "Outputs/Task_1_ICASSP")
    POSITIONS_PATH = os.path.join(DATA_DIR, "Positions")
    RADIATION_PATTERNS_PATH = os.path.join(DATA_DIR, "Radiation_Patterns")
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, args.model_dir)


    inputs_list = []
    for b in range(1, 26):
        for ant in range(1, 3):
            for f in range(1, 4):
                for sp in range(80):
                    inp = f"B{b}_Ant{ant}_f{f}_S{sp}.png"
                    out = inp
                    if not (os.path.exists(os.path.join(INPUT_PATH,  inp)) and
                            os.path.exists(os.path.join(OUTPUT_PATH, out))):
                        continue
                    radar_sample = RadarSampleInputs(
                        ids = (b, ant, f, sp),
                        sampling_position  = sp,
                        freq_MHz = freqs_MHz[f-1],
                        input_file = os.path.join(INPUT_PATH, inp),
                        output_file = os.path.join(OUTPUT_PATH, out),
                        position_file  = os.path.join(POSITIONS_PATH, f"Positions_B{b}_Ant{ant}_f{f}.csv"),
                        radiation_pattern_file = os.path.join(RADIATION_PATTERNS_PATH, f"Ant{ant}_Pattern.csv"),
                    )
                    inputs_list.append(radar_sample)


    logger = TrainingLogger()
    print(f"\nLogging to {logger.log_dir}\n")
    split_path = os.path.join(logger.log_dir, "train_val_split.pkl")
    train_files, val_files = split_data_task1(inputs_list, split_save_path=split_path)
    print(f"Train: {len(train_files)}  |  Val: {len(val_files)}")


    augmentations = AugmentationPipeline([
        RandomWallsAugmentation(p=1.0),
        GeometricAugmentation(
            p=0.5,
            flip_vertical=True,
            flip_horizontal=True,
            cardinal_rotation=True),
    ])

    train_ds = PathlossDataset(train_files, load_output=True, training=True, augmentations=augmentations)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    dummy_net = PathLossNet(in_channels=6, use_selector=args.use_selector)
    optimizer = optim.Adam(dummy_net.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500, verbose=True)

    model_cfg = dict(
        in_channels = 6,
        use_selector= bool(args.use_selector),
        k_frac      = args.k_frac,
        tau         = args.tau
    )

    train_model(
        model_cfg     = model_cfg,
        train_loader  = train_dl,
        val_samples   = val_files,
        optimizer     = optimizer,
        scheduler     = scheduler,
        num_epochs    = args.epochs,
        save_dir      = MODEL_SAVE_PATH,
        logger        = logger,
        device        = device,
        criterion_mode= "sip",          # masked-MSE → "mse"
        entropy_weight= 1e-4)

if __name__ == "__main__":
    main()
