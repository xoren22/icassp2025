import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torchvision.models import resnet18, ResNet18_Weights

from data_module import PathlossNormalizer


class ResNetModel(nn.Module):
    def __init__(self, n_channels=6, n_classes=1, bilinear=True):
        """
        ResNet-18 based model for predicting updates to pathloss map
        
        Args:
            n_channels (int): Number of input channels (default: 5)
            - 4 for environment (reflectance, transmittance, distance, radiation)
            - 1 for current solution
            n_classes (int): Number of output channels (default: 1)
            bilinear (bool): Whether to use bilinear upsampling (default: True)
                             (included for compatibility with ResNetModel interface)
        """
        super(ResNetModel, self).__init__()

        self.normalizer = PathlossNormalizer()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Load pretrained ResNet-18
        resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
        # Modify the first conv layer to accept n_channels
        # Save the weights of the first two layers before modifying
        first_conv_weights = resnet.conv1.weight.data.clone()
        first_bn_weights = {
            'weight': resnet.bn1.weight.data.clone(),
            'bias': resnet.bn1.bias.data.clone(),
            'running_mean': resnet.bn1.running_mean.clone(),
            'running_var': resnet.bn1.running_var.clone()
        }
        
        # Replace the first conv layer to accept n_channels input
        resnet.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize the new conv layer with proper weights
        # For the original 3 channels, use the pretrained weights
        # For the additional channels, initialize with the mean of the pretrained weights
        with torch.no_grad():
            if n_channels >= 3:
                resnet.conv1.weight.data[:, :3, :, :] = first_conv_weights[:, :3, :, :]
                if n_channels > 3:
                    for i in range(3, n_channels):
                        # Initialize additional channels with the mean of RGB weights
                        resnet.conv1.weight.data[:, i:i+1, :, :] = first_conv_weights[:, :3, :, :].mean(dim=1, keepdim=True)
            else:
                # If input channels < 3, use subset of pretrained weights
                resnet.conv1.weight.data[:, :n_channels, :, :] = first_conv_weights[:, :n_channels, :, :]
                
        # Restore the weights of the first batch norm layer
        resnet.bn1.weight.data = first_bn_weights['weight']
        resnet.bn1.bias.data = first_bn_weights['bias']
        resnet.bn1.running_mean = first_bn_weights['running_mean']
        resnet.bn1.running_var = first_bn_weights['running_var']
        
        # Reset weights of all other layers
        for name, module in resnet.named_children():
            if name not in ['conv1', 'bn1']:
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm2d):
                    module.reset_parameters()
        
        # Remove the final fully connected layer and average pooling
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])
        
        # Add a final convolutional layer to get the right number of output channels
        self.final_conv = nn.Conv2d(512, n_classes, kernel_size=1)
        
        # Add upsampling layer to match input spatial dimensions
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear' if bilinear else 'nearest', align_corners=True if bilinear else None)

    def forward(self, x):
        # Extract current solution from input (channel 4)
        current_solution = x[:, 5:6, :, :]
        
        # Pass through ResNet features
        features = self.resnet_features(x)
        
        # Apply final convolution to get the right number of channels
        corrections = self.final_conv(features)
        
        # Upsample back to input resolution
        corrections = self.upsample(corrections)
        
        # Return the updated solution directly (current + correction)
        output = current_solution + corrections

        return output


class ResNetIterative(nn.Module):
    """
    A wrapper that applies iterative refinement to any base model.
    
    The wrapper runs the base model multiple times, adding the current solution
    as the last channel and feeding each prediction back into the input for the next iteration.
    It returns all intermediate predictions which can be used for training with multi-step loss.
    """
    
    def __init__(self, base_model, normalizer, num_iterations=3):
        """
        Initialize the wrapper.
        
        Args:
            base_model: The model to wrap
            num_iterations: Number of refinement iterations to perform
        """
        super(ResNetIterative, self).__init__()
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
    

