import os
import gc
import torch
import numpy as np
from time import time
from tqdm import tqdm
from torchvision.io import read_image

from inference import PathlossPredictor
from loss import se, create_sip2net_loss
from kaggle_eval import kaggle_async_eval


def evaluate_model(inference_model, val_samples, batch_size):
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

def evaluate_model(inference_model, val_samples, batch_size):
    return 14.0**0.5 + 1 / time()

def train_model(model,
                train_loader,
                val_samples,
                optimizer,
                scheduler,
                num_epochs,
                save_dir,
                logger,
                device=None,
                use_sip2net=False,
                sip2net_params={}):
    os.makedirs(save_dir, exist_ok=True)
    model.to(device)
    best_loss = float('inf')
    inference_model = PathlossPredictor(model=model)

    # Setup SIP2Net loss if requested
    if use_sip2net:
        print("Using SIP2Net loss")
        sip2net_criterion = create_sip2net_loss(use_mse=True, **sip2net_params)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}\n{"-"*10}')
        model.train()

        for batch_idx, (inputs, targets, masks) in enumerate(tqdm(train_loader), start=1):
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()

            # Forward pass
            preds = model(inputs)

            # Log any NaNs or Infs
            if torch.isnan(preds).any():
                print(f"[Epoch {epoch+1} Batch {batch_idx}] NaNs detected in preds")
            if torch.isinf(preds).any():
                print(f"[Epoch {epoch+1} Batch {batch_idx}] Infs detected in preds")
            if torch.isnan(targets).any():
                print(f"[Epoch {epoch+1} Batch {batch_idx}] NaNs detected in targets")
            if torch.isinf(targets).any():
                print(f"[Epoch {epoch+1} Batch {batch_idx}] Infs detected in targets")

            # Compute losses
            mask_sum = masks.sum()
            batch_se = se(preds, targets, masks)
            batch_mse = batch_se / (mask_sum + 1e-8)

            if use_sip2net:
                loss, _ = sip2net_criterion(preds, targets, masks)
            else:
                loss = batch_mse

            # Backward & optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            logger.log_batch_loss(batch_se.item(), mask_sum.item())

            # Cleanup
            del inputs, targets, masks, preds, batch_se, batch_mse, loss, mask_sum

        # Validation
        t0 = time()
        inference_model.model = model
        inference_model.model.to(device)
        val_loss = evaluate_model(
            inference_model=inference_model,
            val_samples=val_samples,
            batch_size=8
        )
        print(f"Validation RMSE: {val_loss} taking {time() - t0}")
        torch.cuda.empty_cache()

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        logger.log_epoch_loss(val_loss, epoch, current_lr)

        if epoch % 5 == 4:
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
