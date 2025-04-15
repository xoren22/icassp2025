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

        self.train_batch_se_sum = 0.0
        self.train_batch_mask_sum = 0.0

    def log_batch_loss(self, se_sum, mask_sum):
        self.train_batch_se_sum += se_sum
        self.train_batch_mask_sum += mask_sum

        batch_mse = se_sum / mask_sum
        self.writer.add_scalar("batch_mse", batch_mse, self.global_step)
        self.global_step += 1

    def log_epoch_loss(self, val_epoch_loss, epoch, learning_rate=None):
        epoch_training_loss = (self.train_batch_se_sum / (self.train_batch_mask_sum + 1e-8))**0.5

        self.train_batch_se_sum = 0.0
        self.train_batch_mask_sum = 0

        # Use add_scalars so both series appear on the same chart
        self.writer.add_scalars(
            "epoch_rmse",
            {
                "training": epoch_training_loss,
                "validation": val_epoch_loss
            },
            epoch
        )

        if learning_rate is not None:
            self.writer.add_scalar("lr", learning_rate, epoch)
    
    def log_kaggle_eval(self, kaggle_score, epoch):
        self.writer.add_scalars(
            "epoch_rmse",
            {
                "kaggle": kaggle_score**0.5, # kaggle scores are in mse, we need rmse
            },
            epoch
        )
        self.writer.flush()


    def close(self):
        self.writer.close()
