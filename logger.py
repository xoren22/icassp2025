import os
import torch
from torch.utils.tensorboard import SummaryWriter

class TrainingLogger:
    """Logging utility for tracking training progress using TensorBoard"""
    
    def __init__(self, log_dir='logs'):
        """
        Initialize the logger
        
        Args:
            log_dir: Directory to save TensorBoard logs
        """
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.step = 0
        self.epoch = 0
    
    def log_batch(self, batch_idx, batch_size, batch_loss, running_loss, num_batches):
        """
        Log metrics for a single batch
        
        Args:
            batch_idx: Current batch index
            batch_size: Size of the batch
            batch_loss: Loss for the current batch
            running_loss: Running average loss
            num_batches: Total number of batches in epoch
        """
        # Calculate global step
        self.step = self.epoch * num_batches + batch_idx
        
        # Log batch loss (now RMSE)
        self.writer.add_scalar('Training/BatchRMSE', batch_loss / batch_size, self.step)
        self.writer.add_scalar('Training/RunningAvgRMSE', running_loss / ((batch_idx + 1) * batch_size), self.step)
    
    def log_epoch(self, train_loss, val_loss, learning_rate=None):
        """
        Log metrics for a complete epoch
        
        Args:
            train_loss: Training loss for the epoch
            val_loss: Validation loss for the epoch
            learning_rate: Current learning rate (optional)
        """
        # Log train and validation losses together under one tag for easier comparison
        self.writer.add_scalars('RMSE', {
            'Train': train_loss,
            'Validation': val_loss
        }, self.epoch)
        
        # Log learning rate separately
        if learning_rate is not None:
            self.writer.add_scalar('LearningRate', learning_rate, self.epoch)
        
        # Increment epoch counter
        self.epoch += 1
    
    def log_model(self, model, example_input):
        """
        Log model graph and parameters
        
        Args:
            model: PyTorch model
            example_input: Example input tensor for model visualization
        """
        # Log model graph
        self.writer.add_graph(model, example_input)
        
        # Log parameter histograms
        for name, param in model.named_parameters():
            self.writer.add_histogram(f'Parameters/{name}', param, self.epoch)
    
    def log_image(self, tag, image, step=None):
        """
        Log an image to TensorBoard
        
        Args:
            tag: Image name
            image: Image tensor (C,H,W)
            step: Step to use (defaults to current epoch)
        """
        if step is None:
            step = self.epoch
        
        self.writer.add_image(tag, image, step)
    
    def log_images(self, prefix, input_image, target_image, prediction_image, step=None):
        """
        Log input, target and prediction images
        
        Args:
            prefix: Prefix for the image tags
            input_image: Input image tensor
            target_image: Target image tensor
            prediction_image: Prediction image tensor
            step: Step to use (defaults to current epoch)
        """
        if step is None:
            step = self.epoch
        
        # Log individual images
        self.writer.add_image(f'{prefix}/Input', input_image, step)
        self.writer.add_image(f'{prefix}/Target', target_image, step)
        self.writer.add_image(f'{prefix}/Prediction', prediction_image, step)
    
    def close(self):
        """Close the TensorBoard writer"""
        self.writer.close()