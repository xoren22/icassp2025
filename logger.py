import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from config import LOG_DIR
from config import OUTPUT_SCALER

class TrainingLogger:
    def __init__(self, session_name=None):
        if not session_name:
            session_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = os.path.join(LOG_DIR, session_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.global_step = 0
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.epoch_train_loss_sum = 0.0
        self.epoch_train_steps_sum = 0

    def log_batch_loss(self, loss_values, batch_size):
        val = loss_values
        self.writer.add_scalar("batch_loss", val, self.global_step)
        self.epoch_train_loss_sum += val * batch_size
        self.epoch_train_steps_sum += batch_size
        self.global_step += 1

    def log_epoch_loss(self, val_iteration_loss, epoch, learning_rate=None):
        total_loss_scaled = self.epoch_train_loss_sum * OUTPUT_SCALER
        avg_loss = total_loss_scaled / self.epoch_train_steps_sum if self.epoch_train_steps_sum > 0 else float("inf")

        self.epoch_train_loss_sum = 0.0
        self.epoch_train_steps_sum = 0

        self.writer.add_scalar("epoch_loss_training", avg_loss, epoch)
        self.writer.add_scalar("epoch_loss_validation", val_iteration_loss, epoch)

        if learning_rate is not None:
            self.writer.add_scalar("lr", learning_rate, epoch)

    def close(self):
        self.writer.close()
