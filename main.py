import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import gc

# Import custom modules
from data_module import PathlossDataset, PathlossNormalizer
from model import ResNetModel
from train_module import train_model, evaluate_iterative_model
from visualization import visualize_results
from logger import TrainingLogger

# Import the new iterative model wrapper
from model import ResNetIterative, ResNetModel

def main():
    parser = argparse.ArgumentParser(description='Train and evaluate pathloss prediction model')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID to use (default: auto-select)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=256, help='Image size')
    parser.add_argument('--iterations', type=int, default=5, help='Number of refinement iterations')
    parser.add_argument('--use_mixed_precision', action='store_true', help='Use mixed precision training')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--model_dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for TensorBoard logs')
    parser.add_argument('--plot_dir', type=str, default='plots', help='Directory to save plots')
    
    args = parser.parse_args()
    
    # Set device
    if args.gpu is not None:
        device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Clear CUDA cache at the start
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Define paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, args.data_dir)
    INPUT_PATH = os.path.join(DATA_DIR, "Inputs/Task_3_ICASSP/")
    OUTPUT_PATH = os.path.join(DATA_DIR, "Outputs/Task_3_ICASSP/")
    POSITIONS_PATH = os.path.join(DATA_DIR, "Positions/")
    BUILDING_DETAILS_PATH = os.path.join(DATA_DIR, "Building_Details/")
    RADIATION_PATTERNS_PATH = os.path.join(DATA_DIR, "Radiation_Patterns/")
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, args.model_dir)
    LOG_DIR = os.path.join(BASE_DIR, args.log_dir)
    PLOT_DIR = os.path.join(BASE_DIR, args.plot_dir)
    
    # Create necessary directories
    for dir_path in [MODEL_SAVE_PATH, LOG_DIR, PLOT_DIR]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Initialize logger
    logger = TrainingLogger(log_dir=LOG_DIR)
    
    # Create list of file IDs (b, ant, f, sp)
    file_list = []
    
    # Check if actual data files exist
    data_found = False
    
    # Create file list based on available data
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
                        data_found = True
    
    # Ensure we have at least one entry in each set
    if len(file_list) < 3:
        train_files = file_list[:1]
        val_files = file_list[:1]
        test_files = file_list[:1]
    else:
        # Split data into train, validation, and test sets
        np.random.shuffle(file_list)
        
        # Reduce dataset size if needed to save memory
        max_samples = min(len(file_list), 10_000)  # Limit to 10_000 samples to save memory
        if len(file_list) > max_samples:
            print(f"Limiting dataset to {max_samples} samples to save memory")
            file_list = file_list[:max_samples]
        
        train_size = int(0.7 * len(file_list))
        val_size = int(0.15 * len(file_list))
        
        train_files = file_list[:train_size]
        val_files = file_list[train_size:train_size+val_size]
        test_files = file_list[train_size+val_size:]
    
    print(f"Train: {len(train_files)}, Validation: {len(val_files)}, Test: {len(test_files)}")
    
    # Create datasets
    train_dataset = PathlossDataset(
        train_files, INPUT_PATH, OUTPUT_PATH, POSITIONS_PATH, 
        BUILDING_DETAILS_PATH, RADIATION_PATTERNS_PATH, args.img_size, training=True
    )
    
    val_dataset = PathlossDataset(
        val_files, INPUT_PATH, OUTPUT_PATH, POSITIONS_PATH, 
        BUILDING_DETAILS_PATH, RADIATION_PATTERNS_PATH, args.img_size, training=False
    )
    
    test_dataset = PathlossDataset(
        test_files, INPUT_PATH, OUTPUT_PATH, POSITIONS_PATH, 
        BUILDING_DETAILS_PATH, RADIATION_PATTERNS_PATH, args.img_size, training=False,
    )
    
    # Create data loaders with fewer workers to save memory
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # Initialize base model
    base_model = ResNetModel(n_channels=6, n_classes=1, bilinear=True).to(device)
    
    # Initialize the iterative refinement wrapper model
    model = ResNetIterative(
        base_model, 
        num_iterations=args.iterations,
        normalizer=PathlossNormalizer()
    ).to(device)
    
    # Print model summary
    print(f"ResNetIterative model with {args.iterations} iterations")
    
    # Define loss function and optimizer
    from train_module import RMSELoss
    criterion = RMSELoss()  # Using RMSE instead of MSE
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50, verbose=True)
    
    # Set up mixed precision training if requested
    if args.use_mixed_precision:
        try:
            from torch.cuda.amp import autocast, GradScaler
            print("Using mixed precision training")
            # Continue with setup in train_model
        except ImportError:
            print("Mixed precision training not available, using standard precision")
            args.use_mixed_precision = False
    
    # Train model
    trained_model = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        logger=logger, device=device, num_epochs=args.epochs, save_dir=MODEL_SAVE_PATH
    )
    
    # Evaluate model
    test_loss = evaluate_iterative_model(trained_model, test_loader, criterion, device=device)
    print(f"Test RMSE: {test_loss:.4f}")
    
    # Visualize results (you might need to adjust this function to work with the wrapper)
    # For visualization, you'll need to collect predictions explicitly
    print("Generating visualizations...")
    results = []
    trained_model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Get predictions - taking only the final iteration
            predictions = trained_model(inputs)[:, -1]
            
            # Store results for visualization
            results.append({
                'input': inputs.cpu().numpy(),
                'target': targets.cpu().numpy(),
                'prediction': predictions.cpu().numpy()
            })
    
    visualize_results(results, save_dir=PLOT_DIR)
    
    # Close logger
    logger.close()
    
    print("Training and evaluation complete!")


if __name__ == "__main__":
    # tensorboard --logdir=/auto/home/xoren/icassp2025/logs
    main()