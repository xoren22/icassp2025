import os
import gc
import torch
import numpy as np
from time import time
from tqdm import tqdm
from torchvision.io import read_image
from torch.amp import autocast, GradScaler

from loss import se
from inference import PathlossPredictor
from kaggle_eval import kaggle_async_eval


def evaluate_model(inference_model, val_samples, batch_size=8):
    preds_list, targets_list = [], []
    val_samples = list(val_samples)

    all_inputs, all_targets = [], []
    for sample in val_samples:
        d = sample.asdict()
        target = read_image(d.pop("output_file")).float()
        all_inputs.append(d)
        all_targets.append(target)

    with torch.no_grad():
        for start_idx in tqdm(range(0, len(all_inputs), batch_size), desc="Evaluating validation set"):
            end_idx = start_idx + batch_size
            batch_inputs = all_inputs[start_idx:end_idx]
            batch_targets = all_targets[start_idx:end_idx]

            batch_preds = inference_model.predict(batch_inputs)
            for pred_i, target_i in zip(batch_preds, batch_targets):
                preds_list.extend(pred_i.cpu().numpy().ravel())
                targets_list.extend(target_i.cpu().numpy().ravel())

    preds_np = np.array(preds_list)
    targets_np = np.array(targets_list)
    val_rmse = np.sqrt(np.mean((np.square(preds_np - targets_np))))

    return val_rmse
    

def train_model(model, train_loader, val_samples, optimizer, scheduler, num_epochs, save_dir, logger, device=None, use_sip2net=False, sip2net_params={}):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    best_loss = float('inf')
    scaler = GradScaler(enabled=True)
    inference_model = PathlossPredictor(model=model)
    
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

                mask_sum = masks.sum()
                batch_se = se(preds, targets, masks)
                batch_mse = batch_se / masks.sum()
                
                loss = batch_mse

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            logger.log_batch_loss(batch_se.item(), mask_sum.item())

            del inputs, targets, masks, preds, batch_se, batch_mse, loss, mask_sum

        t0 = time()

        inference_model.model = model
        inference_model.model.to(device)
        val_loss = evaluate_model(inference_model=inference_model, val_samples=val_samples, batch_size=8)
        print(f"Validation RMSE: {val_loss} taking {time() - t0}")

        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            scheduler.step(val_loss)
        logger.log_epoch_loss(val_loss, epoch, current_lr)

        kaggle_async_eval(
            epoch=epoch,
            logger=logger,
            model=inference_model,
        )

        if val_loss is not None and val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f'Saved new best model (Val RMSE: {val_loss:.4f}).')
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'epoch_{epoch}.pth'))

        gc.collect()