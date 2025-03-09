import os
import gc
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.amp import autocast, GradScaler


class RMSELoss(nn.Module):
    def __init__(self, do_tests=False):
        super().__init__()
        self.do_tests = do_tests

    def forward(self, pred, target, mask):
        """
        pred:   [batch, iterations, H, W]
        target: [batch, H, W]
        mask:   [batch, H, W]
        Returns RMSE of shape [batch, iterations].
        """
        if self.do_tests:
            if not torch.is_floating_point(mask):
                raise ValueError("Mask must be floating type for a proper weighted sum.")
            if mask.sum() <= 0:
                raise ValueError("Mask sums to zero; division by zero is imminent.")

        target = target.unsqueeze(1) # Expand target/mask to broadcast over the 'iterations' dimension
        mask = mask.unsqueeze(1)

        sq_error = (pred - target) ** 2 * mask
        sum_sq_error = sq_error.sum(dim=[2, 3])   # shape [batch, iterations]
        sum_mask = mask.sum(dim=[2, 3])           # shape [batch, iterations]

        rmse = torch.sqrt(sum_sq_error / (sum_mask + 1e-8))  # [batch, iterations]
        return rmse


def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    iteration_sum = 0.0  # Will hold sum of RMSEs over the batch dimension
    iteration_count = 0   # How many samples we've summed over

    with torch.no_grad():
        for inputs, targets, masks in data_loader:
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)

            predictions = model(inputs) # [batch, iterations, H, W]
            losses_per_sample_iter = criterion(predictions, targets, masks) # [batch, iterations]
            sum_across_batch = losses_per_sample_iter.sum(dim=0) # summing across the batch => shape [iterations]

            iteration_sum += sum_across_batch
            iteration_count += losses_per_sample_iter.size(0)

    if iteration_count == 0:
        return [None]

    iteration_mean = iteration_sum / iteration_count

    return iteration_mean.tolist()


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, save_dir, logger, device=None):
    os.makedirs(save_dir, exist_ok=True)

    model.to(device)
    best_loss = float('inf')
    scaler = GradScaler(enabled=True)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}\n{"-"*10}')
        model.train()
        running_loss = 0.0

        for batch_idx, (inputs, targets, masks) in enumerate(tqdm(train_loader), start=1):
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)

            optimizer.zero_grad()
            with autocast('cuda', enabled=True):
                all_preds = model(inputs)
                per_sample_iter_loss = criterion(all_preds, targets, masks)
                per_iter_loss = per_sample_iter_loss.mean(0)
                loss = per_iter_loss.mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            logger.log_batch_loss(loss_values=per_iter_loss, batch_size=inputs.size(0))

            del inputs, targets, masks, all_preds
            torch.cuda.empty_cache()

        val_iterations_loss = evaluate_model(model, val_loader, criterion, device=device)
        
        val_final_iteration_loss = val_iterations_loss[-1]
        print(f"Validation RMSE for iterations: {val_iterations_loss}")

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_final_iteration_loss)
        
        logger.log_epoch_loss(val_iterations_loss, epoch, current_lr)

        if val_final_iteration_loss < best_loss:
            best_loss = val_final_iteration_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f'Saved new best model (Val RMSE: {val_final_iteration_loss:.4f}).')

        gc.collect()
        torch.cuda.empty_cache()


