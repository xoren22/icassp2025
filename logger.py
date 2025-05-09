import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from config import LOG_DIR, OUTPUT_SCALER


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
        epoch_training_loss *= OUTPUT_SCALER
        
        self.train_batch_se_sum = 0.0
        self.train_batch_mask_sum = 0

        # Use add_scalars so both series appear on the same chart
        self.writer.add_scalars(
            "epoch_mse",
            {
                "training": epoch_training_loss**2,
                "validation": val_epoch_loss**2,
            },
            epoch
        )

        if learning_rate is not None:
            self.writer.add_scalar("lr", learning_rate, epoch)
    
    def log_kaggle_eval(self, kaggle_score, epoch):
        self.writer.add_scalars(
            "epoch_mse",
            {
                "kaggle": kaggle_score, # kaggle scores are in mse, we need rmse
            },
            epoch
        )
        self.writer.flush()


    def close(self):
        self.writer.close()
