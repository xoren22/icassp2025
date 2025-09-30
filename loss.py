def se(preds, targets, masks):
    sq_error = (preds - targets)**2 * masks
    sq_error_sum = sq_error.sum()
    return sq_error_sum
