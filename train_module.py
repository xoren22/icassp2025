# train.py   (drop-in replacement)

import torchvision
from tqdm import tqdm
import os, gc, time, torch, numpy as np
from torch.cuda.amp import autocast, GradScaler

from model import PathLossNet
from loss  import PathLossCriterion, create_sip2net_loss
from inference import PathlossPredictor
from kaggle_eval import kaggle_async_eval
from loss import se                       # still used for on-the-fly MSE

# ---------------------------------------------------- validation ----------
@torch.no_grad()
def evaluate_model(inference_model, val_samples, batch_size=8, device="cuda"):
    preds_list, targets_list = [], []
    val_samples = list(val_samples)

    all_inputs, all_targets = [], []
    for sample in val_samples:
        d = sample.asdict()
        target = torchvision.io.read_image(d.pop("output_file")).float()
        all_inputs.append(d)
        all_targets.append(target)

    for i in tqdm(range(0, len(all_inputs), batch_size), desc="Validating"):
        batch_in  = all_inputs [i : i + batch_size]
        batch_tgt = all_targets[i : i + batch_size]

        batch_pred = inference_model.predict(batch_in)
        for p, t in zip(batch_pred, batch_tgt):
            preds_list  .extend(p.cpu().numpy().ravel())
            targets_list.extend(t.cpu().numpy().ravel())

    preds_np   = np.asarray(preds_list)
    targets_np = np.asarray(targets_list)
    return np.sqrt(np.mean((preds_np - targets_np) ** 2))


# -------------------------------------------------------- training -------
def train_model(
    model_cfg         : dict,              # kwargs for PathLossNet
    train_loader,
    val_samples,
    optimizer,
    scheduler,
    num_epochs,
    save_dir,
    logger,
    device            = "cuda",
    criterion_mode    = "mse",             # "mse" or "sip"
    entropy_weight    = 1e-4):

    os.makedirs(save_dir, exist_ok=True)
    net = PathLossNet(**model_cfg).to(device)

    # ---- loss -----------------------------------------------------------
    if criterion_mode == "sip":
        sip = create_sip2net_loss(use_mse=True, mse_weight=0.5)
        criterion = PathLossCriterion(sip2net_loss=sip,
                                      entropy_weight=entropy_weight)
    else:
        criterion = PathLossCriterion(sip2net_loss=None, mse_only=True)

    scaler = GradScaler(enabled=True)
    best_loss = float('inf')

    # wrapper for submission-time inference (unchanged)
    inference_model = PathlossPredictor(model=net)

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        net.train()

        for inputs, targets, masks in tqdm(train_loader, desc="Train"):
            inputs, targets, masks = (t.to(device) for t in (inputs, targets, masks))

            # --- choose mode -------------------------------------------
            if net.use_selector:
                fwd_kwargs = dict(y_full=targets)           # probes are learned
            else:
                sparse = targets * masks
                fwd_kwargs = dict(ext_mask=masks, ext_sparse=sparse)

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                out   = net(inputs, **fwd_kwargs)
                loss  = criterion(out, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # simple logging: raw SE and mask sum (unchanged)
            batch_se  = se(out["pred"], targets, masks).item()
            logger.log_batch_loss(batch_se, masks.sum().item())

            del inputs, targets, masks, out, loss

        # ---------------- validation -----------------------------------
        t0 = time.time()
        inference_model.model = net          # sync wrapper
        val_rmse = evaluate_model(inference_model, val_samples,
                                  batch_size=8, device=device)
        print(f"Validation RMSE: {val_rmse:.4f}  (took {time.time()-t0:.1f}s)")

        scheduler.step(val_rmse)
        logger.log_epoch_loss(val_rmse, epoch, optimizer.param_groups[0]['lr'])

        # kaggle_async_eval(epoch=epoch, logger=logger, model=inference_model)

        # save checkpoints
        if val_rmse < best_loss:
            best_loss = val_rmse
            torch.save(net.state_dict(), os.path.join(save_dir, 'best_model.pth'))
            print("✓ saved new best")
        if epoch % 5 == 0:
            torch.save(net.state_dict(), os.path.join(save_dir, f'epoch_{epoch}.pth'))

        gc.collect()
