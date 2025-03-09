import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from config import LOG_DIR

class TrainingLogger:
    def __init__(self, session_name=None):
        if not session_name:
            session_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = os.path.join(LOG_DIR, session_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.global_step = 0
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.epoch_train_loss_sums = []
        self.epoch_train_steps_sums = []

    def log_batch_loss(self, loss_values, batch_size):
        if not self.epoch_train_loss_sums:
            self.epoch_train_loss_sums = [0.0] * len(loss_values)
            self.epoch_train_steps_sums = [0] * len(loss_values)

        for i, val in enumerate(loss_values):
            self.writer.add_scalars("batch_loss", {f"train_iter_{i}": val}, self.global_step)
            self.epoch_train_loss_sums[i] += val * batch_size
            self.epoch_train_steps_sums[i] += batch_size

        self.global_step += 1

    def log_epoch_loss(self, val_iteration_loss, epoch, learning_rate=None):
        train_iteration_losses = []
        for i in range(len(self.epoch_train_loss_sums)):
            total_loss = self.epoch_train_loss_sums[i]
            total_steps = self.epoch_train_steps_sums[i]
            avg_loss = total_loss / total_steps if total_steps > 0 else float("inf")
            train_iteration_losses.append(avg_loss)

        self.epoch_train_loss_sums = []
        self.epoch_train_steps_sums = []

        for i, (tr, vl) in enumerate(zip(train_iteration_losses, val_iteration_loss)):
            self.writer.add_scalars("epoch_loss_iters", {f"training_iter_{i}": tr}, epoch)
            self.writer.add_scalars("epoch_loss_iters", {f"validation_iter_{i}": vl}, epoch)

        self.writer.add_scalars("epoch_loss_final", {f"training": train_iteration_losses[-1]}, epoch)
        self.writer.add_scalars("epoch_loss_final", {f"validation": val_iteration_loss[-1]}, epoch)

        if learning_rate is not None:
            self.writer.add_scalar("lr", learning_rate, epoch)

    def close(self):
        self.writer.close()
