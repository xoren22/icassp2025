import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UNetModel(nn.Module):
    def __init__(self, n_channels=6):
        super(UNetModel, self).__init__()

        self.unet = smp.Unet( # Load pretrained U-Net with ResNet18 encoder, keep pretrained
            classes=1,                      # Output one channel for pathloss prediction
            activation=None,                # No activation - we'll handle this ourselves
            encoder_name="resnet18",        # Use ResNet18 as the encoder backbone
            encoder_weights="imagenet",     # Use pretrained weights on ImageNet
            in_channels=n_channels,         # Match our input channels
        )
        
        
        with torch.no_grad(): # Keep the pretrained weights for the first 3 channels
            if n_channels >= 3:
                original_weights = self.unet.encoder.conv1.weight.data[:, :3, :, :].clone()
                original_conv = self.unet.encoder.conv1
                new_conv = nn.Conv2d(
                    n_channels, 
                    original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                    bias=False if original_conv.bias is None else True
                )

                new_conv.weight.data[:, :3, :, :] = original_weights
                if n_channels > 3:
                    for i in range(3, n_channels):
                        new_conv.weight.data[:, i:i+1, :, :] = original_weights.mean(dim=1, keepdim=True)
                if original_conv.bias is not None:
                    new_conv.bias.data = original_conv.bias.data.clone()
                self.unet.encoder.conv1 = new_conv


    def forward(self, x):
        current_solution = x[:, 5:6, :, :]
        corrections = self.unet(x)
        output = current_solution + corrections

        return output


class UNetIterative(nn.Module):
    def __init__(self, base_model, num_iterations=3, output_scale=160.0):
        super(UNetIterative, self).__init__()
        self.base_model = base_model
        self.output_scale = output_scale
        self.num_iterations = num_iterations
    
    def _rescale_outputs(self, output):
        return (1 + output / 2) * self.output_scale

    def forward(self, x):
        # We'll store each iteration's prediction in all_predictions
        all_predictions = []

        # Repeatedly call base_model, which expects x's last channel to be the current solution
        for i in range(self.num_iterations):
            predicted = self.base_model(x)               # base_model forward uses x[:, -1, :, :] as current_solution
            all_predictions.append(predicted)
            x[:, -1, :, :] = predicted[:, 0, :, :]      # Overwrite x's last channel with the new predicted solution

        output = torch.cat(all_predictions, dim=1)
        if not self.training:
            output = self._rescale_outputs(output)

        return output
