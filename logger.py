import os
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class TrainingLogger:
    """
    A simpler logger for iterative training:
      - Logs final iteration loss each batch (scaled)
      - Logs epoch train/val losses on one plot
      - Logs debug images (input/target/final pred) every N epochs
    """

    def __init__(self, log_root="logs", scale_factor=160.0, image_log_interval=5):
        """
        Args:
            log_root: directory in which to create subdir for new run
            scale_factor: multiply train losses by this factor to match val scale
            image_log_interval: only log debug images every N epochs
        """
        # Put each run in a time-stamped directory, so old logs remain
        time_str = datetime.now().strftime('%Y-%m-%d_%H:%M')
        self.log_dir = os.path.join(log_root, f"run_{time_str}")
        os.makedirs(self.log_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.scale_factor = scale_factor
        self.epoch = 0
        self.global_step = 0
        self.image_log_interval = image_log_interval

        self.debug_samples = None  # (inputs, targets, masks)

    def set_debug_samples(self, inputs, targets, masks):
        """
        Provide a small batch of data (e.g. 5 samples) for visual logging each epoch.
        """
        self.debug_samples = (inputs, targets, masks)

    def log_batch_loss(self, loss_value, batch_size):
        """
        Logs the final iteration loss each batch (train) after scaling.
        """
        scaled_loss = loss_value * self.scale_factor
        # e.g. "Train/BatchRMSE" at self.global_step
        self.writer.add_scalar("Train/BatchRMSE_Scaled", scaled_loss, self.global_step)
        self.global_step += 1

    def log_epoch_loss(self, train_loss, val_loss, learning_rate=None):
        """
        After each epoch, logs scaled train loss & val loss on the same plot.
        """
        scaled_train_loss = train_loss * self.scale_factor
        self.writer.add_scalars("Loss", {
            "TrainScaled": scaled_train_loss,
            "Val": val_loss
        }, self.epoch)

        if learning_rate is not None:
            self.writer.add_scalar("LearningRate", learning_rate, self.epoch)

    def log_debug_images(self, model, device=None):
        """
        Logs input/target/pred for the small debug batch, if set.
        Logs only final iteration predictions from the model.
        Called once per epoch; we skip if epoch not a multiple of self.image_log_interval.
        """
        if self.debug_samples is None:
            return
        
        # Only log every N epochs (default 5)
        if self.epoch % self.image_log_interval != 0:
            return

        model.eval()
        inputs, targets, masks = self.debug_samples
        if device:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

        with torch.no_grad():
            # shape: [num_iterations, B, C, H, W]
            all_preds = model(inputs)
            final_preds = all_preds[-1]  # final iteration, shape [B, C, H, W]

        # We'll log up to 5 images from that batch
        max_images = min(5, final_preds.size(0))
        for i in range(max_images):
            # For clarity, we assume single-channel for target/pred
            inp = inputs[i].cpu()
            tgt = targets[i].cpu()
            pred = final_preds[i].cpu()

            # If your input has many channels, pick or transform them as needed.
            # Here, let's just visualize the first 3 channels of input (if available).
            self.writer.add_image(f"Debug/Epoch_{self.epoch}/Input_{i}",
                                  inp[:3], self.epoch) 
            self.writer.add_image(f"Debug/Epoch_{self.epoch}/Target_{i}",
                                  tgt, self.epoch)
            self.writer.add_image(f"Debug/Epoch_{self.epoch}/Pred_{i}",
                                  pred, self.epoch)

    def on_epoch_end(self):
        """
        Called at end of each epoch to increment epoch count
        """
        self.epoch += 1

    def close(self):
        self.writer.close()
