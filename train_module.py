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


def evaluate_model(model, val_samples, device, batch_size=8, inference_model=None):
    inference_model.model = model
    inference_model.model.to(device)

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

def evaluate_model(model, val_samples, device, batch_size=8, inference_model=None):
    return 1/time()


def train_model(model, train_loader, val_samples, optimizer, scheduler, num_epochs, save_dir, logger, device=None, use_sip2net=False, sip2net_params=None):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    best_loss = float('inf')
    scaler = GradScaler(enabled=True)
    
    # Create a single predictor instance that will be reused for validation
    inference_model = PathlossPredictor(model=model)
    
    # Setup SIP2Net loss if requested
    if use_sip2net:
        if sip2net_params is None:
            sip2net_params = {}
        sip2net_criterion = create_sip2net_loss(
            use_mse=True,
            mse_weight=sip2net_params.get('mse_weight', 1.0),
            ssim_alpha=sip2net_params.get('ssim_alpha', 500.0),
            gdl_alpha=sip2net_params.get('gdl_alpha', 1.0),
            f1_alpha=sip2net_params.get('f1_alpha', 0.0)
        )
        print(f"Using SIP2Net loss")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}\n{"-"*10}')
        model.train()

        for batch_idx, (inputs, targets, masks, wall_density_alphas) in enumerate(tqdm(train_loader), start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            wall_density_alphas = wall_density_alphas.to(device)

            optimizer.zero_grad()
            with autocast('cuda'):
                preds = model(inputs)

                mask_sum = masks.sum()
                batch_se = se(preds, targets, masks, wall_density_alphas)
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
            # torch.cuda.empty_cache()

        t0 = time()

        val_loss = evaluate_model(model, val_samples, device=device, inference_model=inference_model)
        print(f"Validation RMSE: {val_loss} taking {time() - t0}")

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        logger.log_epoch_loss(val_loss, epoch, current_lr)

        if val_loss is not None and val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print(f'Saved new best model (Val RMSE: {val_loss:.4f}).')

        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'epoch_{epoch}.pth'))

        gc.collect()
        # torch.cuda.empty_cache()

