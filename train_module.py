import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import gc
from torch.amp import autocast, GradScaler

# Set environment variable for expandable segments
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch

class RMSELoss(nn.Module):
    """Root Mean Square Error loss function with masking."""
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, pred, target, mask):
        squared_error = (pred - target)**2

        squared_error = squared_error * mask
        mse = squared_error.sum() / mask.sum()

        return torch.sqrt(mse)


def evaluate_iterative_model(model, data_loader, criterion=None, device=None):
    """
    Evaluate a wrapped model on a dataset using only the final prediction.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model.eval()
    total_loss = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for inputs, targets, masks in data_loader:
            inputs = inputs.to(device)
            masks = masks.to(device)
            targets = targets.to(device)
            
            # Forward pass with return_all_iterations=False to get only the final prediction
            predictions = model(inputs)[-1]
            
            # Calculate loss only on the final prediction
            if criterion is None:
                # Default to RMSE if no criterion provided
                loss = torch.sqrt(torch.mean((predictions - targets)**2))
            else:
                loss = criterion(predictions, targets, masks)
            
            # Accumulate loss
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            num_samples += batch_size
            
            # Clear memory
            del inputs, targets, predictions
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return total_loss / num_samples if num_samples > 0 else float('inf')


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                logger=None, device=None, num_epochs=25, save_dir='models/'):
    """
    Train the ResNetIterative model
    """
    if device is None:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    os.makedirs(save_dir, exist_ok=True)
    
    best_loss = float('inf')
    scaler = GradScaler("cuda")  # for mixed precision
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        try:
            batch_count = len(train_loader)
            log_interval = max(1, batch_count // 10)
            
            for batch_idx, (inputs, targets, mask) in enumerate(tqdm(train_loader)):
                inputs = inputs.to(device)
                mask = mask.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                
                with autocast(device_type="cuda"):
                    all_predictions = model(inputs)
                    
                    # Loss across all iterations
                    loss = 0
                    for i in range(all_predictions.shape[0]):
                        step_loss = criterion(all_predictions[i], targets, mask)
                        loss += step_loss
                    loss = loss / all_predictions.shape[0]
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # For logging, we consider the final iteration loss as "batch_loss":
                final_loss = criterion(all_predictions[-1], targets, mask).item()
                
                batch_loss = loss.item() * inputs.size(0)  # sum of entire batch
                running_loss += batch_loss
                
                # Log to TensorBoard, if available
                if logger is not None:
                    # 1) log the final iteration's RMSE
                    logger.log_batch_loss(final_loss, inputs.size(0))
                
                # Print intermediate info
                if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == batch_count:
                    current_loss = batch_loss / inputs.size(0)
                    avg_loss = running_loss / ((batch_idx + 1) * inputs.size(0))
                    print(f'  Batch {batch_idx + 1}/{batch_count} - RMSE: {current_loss:.4f}, Avg RMSE: {avg_loss:.4f}')
                    print(f'  Final Iteration RMSE: {final_loss:.4f}')
                
                del inputs, targets, all_predictions
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Exception during training: {e}")
            import traceback
            traceback.print_exc()
            torch.cuda.empty_cache()
            gc.collect()
            continue
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation phase
        print("Validating...")
        val_loss = evaluate_iterative_model(model, val_loader, criterion, device=device)
        print(f"Validation RMSE: {val_loss:.4f}")
        
        # Update LR
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # Log to TensorBoard if logger is provided
        if logger is not None:
            logger.log_epoch_loss(epoch_loss, val_loss, current_lr)
            logger.log_debug_images(model, device=device)  # log images if debug samples set
            logger.on_epoch_end()
        
        print(f'Epoch {epoch+1} summary - Train RMSE: {epoch_loss:.4f}, Val RMSE: {val_loss:.4f}, LR: {current_lr:.6f}')
        
        # Save model if improved
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f'Saved model with Val RMSE: {val_loss:.4f}')
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))
            print(f'Saved model checkpoint at epoch {epoch+1}')
        
        gc.collect()
        torch.cuda.empty_cache()
    
    # Attempt to reload best model
    try:
        model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    except Exception as e:
        print(f"Could not load best model, using current model instead: {e}")
    
    return model
