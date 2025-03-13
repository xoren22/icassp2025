import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

from config import OUTPUT_SCALER


class UNetModel(nn.Module):
    def __init__(self, n_channels=6, output_scale=OUTPUT_SCALER):
        super(UNetModel, self).__init__()

        self.output_scale = output_scale
        self.unet = smp.Unet(
            encoder_name="resnet18",
            encoder_weights="imagenet",
            in_channels=n_channels,
            classes=1,
            activation=None  # We'll produce raw logits
        )

        with torch.no_grad():
            # If n_channels <= 3, you can leave the encoder as-is.
            # If n_channels > 3, replicate the pretrained conv1 weights for extra channels.
            if n_channels > 3:
                original_conv = self.unet.encoder.conv1
                original_weights = original_conv.weight.data[:, :3, :, :].clone()

                new_conv = nn.Conv2d(
                    in_channels=n_channels,
                    out_channels=original_conv.out_channels,
                    kernel_size=original_conv.kernel_size,
                    stride=original_conv.stride,
                    padding=original_conv.padding,
                    bias=(original_conv.bias is not None)
                )

                # Copy the pretrained weights for the first 3 channels
                new_conv.weight.data[:, :3, :, :] = original_weights

                # For any extra channels, replicate the average of those first 3 channels
                for i in range(3, n_channels):
                    new_conv.weight.data[:, i:i+1, :, :] = original_weights.mean(dim=1, keepdim=True)

                if original_conv.bias is not None:
                    new_conv.bias.data = original_conv.bias.data.clone()

                self.unet.encoder.conv1 = new_conv

    def _rescale_outputs(self, output):
        return (1 + output / 2) * self.output_scale

    def forward(self, x):
        output = self.unet(x).squeeze(1)
        if not self.training:
            output = self._rescale_outputs(output)
        return output
