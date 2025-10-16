import os
import gc
import torch
import numpy as np
from time import time
from tqdm import tqdm
from torchvision.io import read_image
from torch.amp import autocast, GradScaler
from memory_hogger import MemoryHogger

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

    # GPU memory reservation: keep almost all VRAM reserved and free just-in-time headroom
    hog = None
    if device is not None and isinstance(device, torch.device) and device.type == 'cuda':
        hog = MemoryHogger(device=str(device), base_headroom_mib=512, chunk_mib=32)
        hog.reserve()

    # Adaptive budgets per phase
    budgets = {"train_step": None, "val_epoch": None}
    safety_bytes = 1024 * 1024 * 1024
    initial_headroom = 2 * 1024 * 1024 * 1024

    # Optional warmup: one dry train step and a short eval to record peaks
    # This helps set tighter budgets before the first epoch
    if hog is not None:
        try:
            # Try grabbing one batch for warmup if available
            first_batch = next(iter(train_loader))
            inputs, targets, masks = first_batch
            torch.cuda.reset_peak_memory_stats(device)
            with hog.phase(initial_headroom + safety_bytes):
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
                loss.backward()  # unscaled warmup backward
                del inputs, targets, masks, preds, batch_se, batch_mse, loss, mask_sum
            peak_b = torch.cuda.max_memory_reserved(device)
            budgets["train_step"] = max(budgets["train_step"] or 0, int(peak_b))

            torch.cuda.reset_peak_memory_stats(device)
            with hog.phase(initial_headroom + safety_bytes):
                _ = evaluate_model(inference_model=PathlossPredictor(model=model), val_samples=val_samples, batch_size=4)
            peak_b = torch.cuda.max_memory_reserved(device)
            budgets["val_epoch"] = max(budgets["val_epoch"] or 0, int(peak_b))
        except StopIteration:
            pass
        except RuntimeError:
            # If warmup OOMs, budgets will adapt during training
            pass
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}\n{"-"*10}')
        model.train()

        for batch_idx, (inputs, targets, masks) in enumerate(tqdm(train_loader), start=1):
            # Determine headroom for this step
            headroom = budgets["train_step"] if budgets["train_step"] is not None else initial_headroom
            retry_count = 0
            max_retries = 3

            while True:
                try:
                    if hog is not None:
                        torch.cuda.reset_peak_memory_stats(device)
                        with hog.phase(headroom + safety_bytes):
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

                        # update budget using observed peak
                        peak_b = torch.cuda.max_memory_reserved(device)
                        budgets["train_step"] = max(budgets["train_step"] or 0, int(peak_b))
                    else:
                        # CPU or no hogging: run as before
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

                    break
                except RuntimeError as e:
                    # Handle OOM by expanding headroom and retrying up to max_retries
                    if ("out of memory" in str(e).lower()) and hog is not None and (retry_count < max_retries):
                        retry_count += 1
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        headroom = max(int(headroom * 2), headroom + (512 * 1024 * 1024))
                        continue
                    raise

        t0 = time()

        inference_model.model = model
        inference_model.model.to(device)

        # Validation epoch under a single hogger phase with adaptive headroom
        if hog is not None:
            headroom = budgets["val_epoch"] if budgets["val_epoch"] is not None else initial_headroom
            torch.cuda.reset_peak_memory_stats(device)
            with hog.phase(headroom + safety_bytes):
                val_loss = evaluate_model(inference_model=inference_model, val_samples=val_samples, batch_size=8)
            peak_b = torch.cuda.max_memory_reserved(device)
            budgets["val_epoch"] = max(budgets["val_epoch"] or 0, int(peak_b))
        else:
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

    if hog is not None:
        hog.release_all()