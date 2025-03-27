import os
import gc
import torch
import numpy as np
from time import time
from tqdm import tqdm
from torchvision.io import read_image
from torch.amp import autocast, GradScaler

from inference import PathlossPredictor
from loss import se, create_sip2net_loss


def evaluate_model(model, val_samples, logger, device, use_sip2net=False):
    inference_model = PathlossPredictor(model=model)

    preds_list, masks_list, targets_list = [], [], []

    with torch.no_grad():
        for i, sample in tqdm(enumerate(val_samples), "Evaluating validation set: "):
            sample = sample.asdict()
            target = read_image(sample.pop("output_file")).float()
            pred = inference_model.predict(sample)
            if i < 3:
                logger.writer.add_image(f"validation/sample_{i}_target", target, logger.global_step)
                logger.writer.add_image(f"validation/sample_{i}_prediction", pred[None, :, :], logger.global_step)

            preds_list += list(pred.cpu().numpy().flatten())
            targets_list += list(target.cpu().numpy().flatten())

    preds_np, targets_np = np.array(preds_list), np.array(targets_list)
    val_rmse = np.sqrt(np.mean(np.square(preds_np - targets_np)))

    return val_rmse

def train_model(model, train_loader, val_samples, optimizer, scheduler, num_epochs, save_dir, logger, device=None, use_sip2net=False, sip2net_params=None):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    best_loss = float('inf')
    scaler = GradScaler(enabled=True)
    
    # Setup SIP2Net loss if requested
    if use_sip2net:
        if sip2net_params is None:
            sip2net_params = {}
        sip2net_criterion = create_sip2net_loss(
            use_mse=True,
            mse_weight=sip2net_params.get('mse_weight', 1.0),
            alpha1=sip2net_params.get('alpha1', 500.0),
            alpha2=sip2net_params.get('alpha2', 1.0),
            alpha3=sip2net_params.get('alpha3', 0.0)
        )
        print(f"Using SIP2Net loss")

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
                
                # Use SIP2Net loss if requested
                if use_sip2net:
                    loss, _ = sip2net_criterion(preds, targets, masks)
                else:
                    loss = batch_mse

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            logger.log_batch_loss(batch_se.item(), mask_sum.item())

            del inputs, targets, masks, preds, batch_se, batch_mse, loss, mask_sum
            torch.cuda.empty_cache()

        t0 = time()
        val_loss = evaluate_model(model, val_samples, logger=logger, device=device, use_sip2net=use_sip2net)
        print(f"Validation RMSE: {val_loss} taking {time() - t0}")

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