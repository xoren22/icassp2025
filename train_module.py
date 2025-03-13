import os
import gc
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.amp import autocast, GradScaler


def evaluate_model(model, data_loader, device):
    model.eval()
    se_sum = 0.0
    mask_sum = 0.0

    with torch.no_grad():
        for inputs, targets, masks in data_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            preds = model(inputs)
            if preds.dim() == 4 and preds.size(1) == 1:
                preds = preds.squeeze(1)

            sq_error = (preds - targets)**2 * masks
            se_sum += sq_error.sum().item()
            mask_sum += masks.sum().item()

    if mask_sum == 0:
        return None

    return (se_sum / (mask_sum + 1e-8))**0.5


def train_model(model, train_loader, val_loader, optimizer, scheduler, num_epochs, save_dir, logger, device=None):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    best_loss = float('inf')
    scaler = GradScaler(enabled=True)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}\n{"-"*10}')
        model.train()

        for batch_idx, (inputs, targets, masks) in enumerate(tqdm(train_loader), start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            with autocast('cuda', enabled=True):
                preds = model(inputs)
                if preds.dim() == 4 and preds.size(1) == 1:
                    preds = preds.squeeze(1)

                sq_error = (preds - targets)**2 * masks
                se_sum = sq_error.sum()
                mask_sum = masks.sum()

                batch_mse = se_sum / (mask_sum + 1e-8)

            scaler.scale(batch_mse).backward()
            scaler.step(optimizer)
            scaler.update()

            logger.log_batch_loss(se_sum.item(), mask_sum.item())

            del inputs, targets, masks, preds, sq_error
            torch.cuda.empty_cache()

        val_loss = evaluate_model(model, val_loader, device=device)
        print(f"Validation RMSE: {val_loss}")

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        logger.log_epoch_loss(val_loss, epoch, current_lr)

        if val_loss is not None and val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f'Saved new best model (Val RMSE: {val_loss:.4f}).')

        gc.collect()
        torch.cuda.empty_cache()
