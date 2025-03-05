import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.cuda.amp import autocast

from data_module import PathlossNormalizer


class UNetModel(nn.Module):
    def __init__(self, n_channels=7, n_classes=1):
        """
        U-Net based model for predicting updates to pathloss map
        
        Args:
            n_channels (int): Number of input channels (default: 6)
            - 5 for environment (reflectance, transmittance, distance, radiation, etc.)
            - 1 for current solution
            n_classes (int): Number of output channels (default: 1)
            bilinear (bool): Whether to use bilinear upsampling (default: True)
        """
        super(UNetModel, self).__init__()

        self.normalizer = PathlossNormalizer()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Load pretrained U-Net with ResNet18 encoder for better feature extraction
        # We keep pretrained weights for the encoder to leverage transfer learning
        self.unet = smp.Unet(
            encoder_name="resnet18",        # Use ResNet18 as the encoder backbone
            encoder_weights="imagenet",     # Use pretrained weights on ImageNet
            in_channels=n_channels,         # Match our input channels
            classes=n_classes,              # Output one channel for pathloss prediction
            activation=None                 # No activation - we'll handle this ourselves
        )
        
        # Initialize the first conv layer to handle our custom channel count
        # Keep the pretrained weights for the first 3 channels if available
        if hasattr(self.unet.encoder, 'conv1') and hasattr(self.unet.encoder.conv1, 'weight'):
            with torch.no_grad():
                if n_channels >= 3:
                    # Save the original weights for the first three channels
                    original_weights = self.unet.encoder.conv1.weight.data[:, :3, :, :].clone()
                    
                    # Create a new conv layer with the right number of input channels
                    original_conv = self.unet.encoder.conv1
                    new_conv = nn.Conv2d(
                        n_channels, 
                        original_conv.out_channels,
                        kernel_size=original_conv.kernel_size,
                        stride=original_conv.stride,
                        padding=original_conv.padding,
                        bias=False if original_conv.bias is None else True
                    )
                    
                    # Copy the weights for the first three channels
                    new_conv.weight.data[:, :3, :, :] = original_weights
                    
                    # Initialize the additional channels with the mean of the RGB weights
                    if n_channels > 3:
                        for i in range(3, n_channels):
                            new_conv.weight.data[:, i:i+1, :, :] = original_weights.mean(dim=1, keepdim=True)
                    
                    # If there was a bias, copy it too
                    if original_conv.bias is not None:
                        new_conv.bias.data = original_conv.bias.data.clone()
                    
                    # Replace the first conv layer
                    self.unet.encoder.conv1 = new_conv

    def forward(self, x):
        # Extract current solution from input (last channel)
        current_solution = x[:, 5:6, :, :]
        
        # Pass through U-Net to get corrections
        corrections = self.unet(x)
        
        # Return the updated solution (current + correction)
        output = current_solution + corrections

        return output


class UNetIterative(nn.Module):
    """
    A wrapper that applies iterative refinement to the U-Net model.
    
    The wrapper runs the base model multiple times, adding the current solution
    as the last channel and feeding each prediction back into the input for the next iteration.
    It returns all intermediate predictions which can be used for training with multi-step loss.
    """
    
    def __init__(self, base_model, normalizer, num_iterations=3):
        """
        Initialize the wrapper.
        
        Args:
            base_model: The model to wrap
            normalizer: The normalizer to use for denormalizing outputs during inference
            num_iterations: Number of refinement iterations to perform
        """
        super(UNetIterative, self).__init__()
        self.base_model = base_model
        self.normalizer = normalizer
        self.num_iterations = num_iterations
    
    def forward(self, x, return_all_iterations=True):
        """
        Forward pass with iterative refinement.
        
        Args:
            x: Input tensor without the current solution channel
            return_all_iterations: Whether to return all intermediate predictions
                                  (True for training, False for inference)
            
        Returns:
            If return_all_iterations=True:
                Tensor of shape (num_iterations, batch_size, channels, height, width)
                containing all intermediate predictions.
            If return_all_iterations=False:
                Only the final prediction with shape (batch_size, channels, height, width)
        """
        batch_size, channels, height, width = x.shape
        
        # Create a tensor for the current solution (initially zeros)
        current_solution = torch.zeros(batch_size, 1, height, width, device=x.device)
        
        # Store all intermediate predictions if needed
        if return_all_iterations:
            all_predictions = []
        
        # Perform iterative refinement
        for i in range(self.num_iterations):
            # Append the current solution as the last channel
            x_with_solution = torch.cat([x, current_solution], dim=1)
            
            # Forward pass through base model
            predicted = self.base_model(x_with_solution)
            
            # Store prediction if needed
            if return_all_iterations:
                all_predictions.append(predicted)
            
            # Update current solution for next iteration
            current_solution = predicted
        
        # Return all predictions or just the final one
        if return_all_iterations:
            # Stack along a new dimension at position 0
            output = torch.stack(all_predictions, dim=0)
        else:
            # Return only the final prediction
            output = current_solution
        
        if not self.training:
            output = self.normalizer.denormalize_output(output)
    
        return output