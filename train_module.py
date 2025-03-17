import os
import gc
import torch
from tqdm import tqdm
from torchvision.utils import make_grid
from torch.amp import autocast, GradScaler

from loss import mse, rmse, se

def evaluate_model(model, val_loader, logger, device):
    model.eval()

    preds_list, masks_list, targets_list = [], [], []

    with torch.no_grad():
        for inputs, targets, masks in val_loader:
            inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)

            preds = model(inputs)
            preds_list.append(preds.cpu())
            targets_list.append(targets.cpu())
            masks_list.append(masks.cpu())

    preds_all = torch.cat(preds_list, dim=0)
    masks_all = torch.cat(masks_list, dim=0)
    targets_all = torch.cat(targets_list, dim=0)

    val_rmse = rmse(preds_all, targets_all, masks_all).item()

    with torch.no_grad():
        for i in [0, 1, 2]:
            inp, tgt, msk = val_loader.dataset[i]  # single sample
            inp = inp.unsqueeze(0).to(device)
            pred = model(inp)
            tgt_cpu = tgt.cpu() / 255 # normalizing to get into range 0,1 
            pred_cpu = pred.cpu().squeeze(0) / 255

            logger.writer.add_image(f"validation/sample_{i}_prediction", pred_cpu.unsqueeze(0), logger.global_step)
            logger.writer.add_image(f"validation/sample_{i}_target", tgt_cpu.unsqueeze(0), logger.global_step)


    return val_rmse


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
            with autocast('cuda'):
                preds = model(inputs)
                if preds.dim() == 4 and preds.size(1) == 1:
                    preds = preds.squeeze(1)

                mask_sum = masks.sum()
                batch_se = se(preds, targets, masks)

                batch_mse = batch_se / masks.sum()

            scaler.scale(batch_mse).backward()
            scaler.step(optimizer)
            scaler.update()

            logger.log_batch_loss(batch_se.item(), mask_sum.item())

            del inputs, targets, masks, preds
            torch.cuda.empty_cache()

        val_loss = evaluate_model(model, val_loader, logger=logger, device=device)
        print(f"Validation RMSE: {val_loss}")

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        logger.log_epoch_loss(val_loss, epoch, current_lr)

        # Checkpoint best model
        if val_loss is not None and val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f'Saved new best model (Val RMSE: {val_loss:.4f}).')

        gc.collect()
        torch.cuda.empty_cache()
