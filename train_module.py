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
    
    Args:
        model: The wrapped model to evaluate
        data_loader: DataLoader for evaluation data
        criterion: Loss function (if None, RMSE will be used)
        device: Torch device to use
        
    Returns:
        Average loss over the dataset
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
    
    # Return average loss over all samples
    return total_loss / num_samples if num_samples > 0 else float('inf')




def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                logger=None, device=None, num_epochs=25, save_dir='models/'):
    """
    Train the ResNetIterative model
    
    Args:
        model: The ResNetIterative model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        logger: TensorBoard logger (optional)
        device: Torch device to use
        num_epochs: Number of epochs to train for
        save_dir: Directory to save model checkpoints
    
    Returns:
        The trained model
    """
    if device is None:
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    best_loss = float('inf')
    
    # Create a GradScaler for mixed precision training
    scaler = GradScaler("cuda")
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        try:
            # Set up progress logging
            batch_count = len(train_loader)
            log_interval = max(1, batch_count // 10)  # Log approximately 10 times per epoch
            
            for batch_idx, (inputs, targets, mask) in enumerate(tqdm(train_loader)):
                inputs = inputs.to(device)
                mask = mask.to(device)
                targets = targets.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass with mixed precision
                with autocast(device_type="cuda"):
                    # Forward pass gets all iterations of predictions
                    all_predictions = model(inputs)
                    
                    # Compute loss across all iterations (averaging happens inside)
                    loss = 0
                    # Calculate loss for each iteration step
                    for i in range(all_predictions.shape[0]):
                        step_loss = criterion(all_predictions[i], targets, mask)
                        loss += step_loss
                    
                    # Average the loss over all iterations
                    loss = loss / all_predictions.shape[0]
                
                # Backward pass and optimize with scaler
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Update running loss
                batch_loss = loss.item() * inputs.size(0)
                running_loss += batch_loss
                
                # Log to TensorBoard if logger is provided
                if logger is not None:
                    logger.log_batch(batch_idx, inputs.size(0), batch_loss, 
                                    running_loss, batch_count)
                
                # Print intermediate loss info
                if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == batch_count:
                    current_loss = batch_loss / inputs.size(0)
                    avg_loss = running_loss / ((batch_idx + 1) * inputs.size(0))
                    print(f'  Batch {batch_idx + 1}/{batch_count} - RMSE: {current_loss:.4f}, Avg RMSE: {avg_loss:.4f}')
                
                # Log the final iteration loss separately for monitoring
                final_loss = criterion(all_predictions[-1], targets, mask).item()
                if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == batch_count:
                    print(f'  Final Iteration RMSE: {final_loss:.4f}')
                
                # Clear some memory
                del inputs, targets, all_predictions
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Exception during training: {e}")
            import traceback
            traceback.print_exc()
            # Clear memory and try to continue to next epoch
            torch.cuda.empty_cache()
            gc.collect()
            continue
        
        epoch_loss = running_loss / len(train_loader.dataset)
        
        # Validation phase
        print("Validating...")
        val_loss = evaluate_iterative_model(model, val_loader, criterion, device=device)
        print(f"Validation RMSE: {val_loss:.4f}")
        
        # Update learning rate scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        
        # Log to TensorBoard if logger is provided
        if logger is not None:
            logger.log_epoch(epoch_loss, val_loss, current_lr)
        
        print(f'Epoch {epoch+1} summary - Train RMSE: {epoch_loss:.4f}, Val RMSE: {val_loss:.4f}, LR: {current_lr:.6f}')
        
        # Save model if validation loss improves
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f'Saved model with Val RMSE: {val_loss:.4f}')
        
        # Save model at regular intervals regardless of validation loss
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_epoch_{epoch+1}.pth'))
            print(f'Saved model checkpoint at epoch {epoch+1}')
            
        # Collect garbage to free memory
        gc.collect()
        torch.cuda.empty_cache()
    
    # Try to load best model
    try:
        model.load_state_dict(torch.load(os.path.join(save_dir, 'best_model.pth')))
    except Exception as e:
        print(f"Could not load best model, using current model instead: {e}")
    
    return model
