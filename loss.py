import torch

def se(preds, targets, masks):
    sq_error = (preds - targets)**2 * masks
    sq_error_sum = sq_error.sum()
    return sq_error_sum

def mse(preds, targets, masks):
    mask_sum = masks.sum()
    sq_error_sum = se(preds, targets, masks)
    return sq_error_sum / (mask_sum + 1e-8)

def rmse(preds, targets, masks):
    mse_loss = mse(preds, targets, masks)
    return torch.sqrt(mse_loss)

def anchored_mse(preds, targets, masks, anchor, alpha=0.1):
    main_loss = mse(preds, targets, masks)
    anchor_loss = mse(preds, anchor, masks)

    weighted_loss = main_loss + alpha * anchor_loss
    return weighted_loss