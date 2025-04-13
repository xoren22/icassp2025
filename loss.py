import torch
import torch.nn as nn
import torch.nn.functional as F
import piq
import segmentation_models_pytorch as smp


def se(preds, targets, masks, wall_density_alpha=1):
    sq_error = (preds - targets)**2 * masks 
    sq_error = sq_error * wall_density_alpha.unsqueeze(1).unsqueeze(1)
    sq_error_sum = sq_error.sum()
    
    return sq_error_sum


class CustomGradientDifferenceLoss(nn.Module):
    """
    Gradient Difference Loss implementation
    (since it's not commonly available in libraries)
    """
    def __init__(self, alpha=1):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred, target, mask=None):
        # Calculate gradients
        pred_dx = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        pred_dy = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        
        target_dx = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        target_dy = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        
        # Calculate gradient differences
        grad_diff_x = torch.abs(pred_dx - target_dx) ** self.alpha
        grad_diff_y = torch.abs(pred_dy - target_dy) ** self.alpha
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            mask_x = mask[:, :, 1:, :]
            mask_y = mask[:, :, :, 1:]
            
            grad_diff_x = grad_diff_x * mask_x
            grad_diff_y = grad_diff_y * mask_y
            
            # Sum and normalize by mask
            loss = (grad_diff_x.sum() / (mask_x.sum() + 1e-8)) + (grad_diff_y.sum() / (mask_y.sum() + 1e-8))
        else:
            loss = torch.mean(grad_diff_x) + torch.mean(grad_diff_y)
            
        return loss


class CustomF1Loss(nn.Module):
    """
    Custom F1 Loss that works correctly with regression problems
    """
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, pred, target, mask=None):
        # For regression problems, we use a different approach than the standard Dice/F1
        # Calculate the "similarity" between predictions and targets
        
        # Calculate absolute difference
        diff = torch.abs(pred - target)
        
        # Scale by the max value for normalization
        max_val = torch.max(torch.max(pred), torch.max(target)) + self.epsilon
        similarity = 1.0 - (diff / max_val)
        
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            similarity = similarity * mask
            denominator = mask.sum() + self.epsilon
        else:
            denominator = torch.numel(pred) + self.epsilon
        
        # Mean similarity is analogous to F1 score for regression
        f1 = similarity.sum() / denominator
        
        # Return 1 - f1 as the loss (0 = perfect match)
        return 1.0 - f1


class SIP2NetLoss(nn.Module):
    """
    SIP2Net loss using library implementations where available
    """
    def __init__(self, ssim_alpha=500, gdl_alpha=1, f1_alpha=1, use_mse=True, mse_weight=1.0):
        super().__init__()
        
        # Use PIQ for SSIM (higher quality implementation)
        self.ssim_loss = piq.SSIMLoss(data_range=255.0)
            
        # GDL is not commonly available, so use custom implementation
        self.gdl_loss = CustomGradientDifferenceLoss(alpha=1)
        
        # Use a custom F1 loss that works better for regression
        self.f1_loss = CustomF1Loss()
            
        self.ssim_alpha = ssim_alpha
        self.gdl_alpha = gdl_alpha
        self.f1_alpha = f1_alpha
        self.use_mse = use_mse
        self.mse_weight = mse_weight
    
    def forward(self, pred, target, mask=None):
        # Ensure inputs have channel dimension
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        if mask is not None and mask.dim() == 3:
            mask = mask.unsqueeze(1)
            
        # Calculate SSIM loss
        # PIQ expects normalized inputs
        pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        target_norm = (target - target.min()) / (target.max() - target.min() + 1e-8)
        
        if mask is not None:
            # Apply mask before calling
            pred_norm = pred_norm * mask
            target_norm = target_norm * mask
            
        ssim = self.ssim_loss(pred_norm, target_norm)
        
        # Calculate GDL loss
        gdl = self.gdl_loss(pred, target, mask)
        
        # Calculate F1 loss (using our custom implementation)
        f1 = self.f1_loss(pred, target, mask)
        
        # Combine losses
        sip2net_loss = self.ssim_alpha * ssim + self.gdl_alpha * gdl + self.f1_alpha * f1
        
        # Add MSE if requested
        if self.use_mse and mask is not None:
            mse = ((pred - target) ** 2 * mask).sum() / (mask.sum() + 1e-8)
            total_loss = sip2net_loss + self.mse_weight * mse
            mse_value = mse.item()
        else:
            total_loss = sip2net_loss
            mse_value = 0.0
            
        # Return total loss and components
        components = {
            'ssim_loss': ssim.item(),
            'gdl_loss': gdl.item(),
            'f1_loss': f1.item(),
            'sip2net_loss': sip2net_loss.item(),
            'mse': mse_value,
            'total_loss': total_loss.item()
        }
        
        return total_loss, components


def create_sip2net_loss(use_mse=True, mse_weight=0.5, ssim_alpha=500, gdl_alpha=1, f1_alpha=1):
    """
    Create a SIP2Net loss instance with specified parameters
    
    Args:
        use_mse: Whether to include MSE in the total loss
        mse_weight: Weight of MSE component relative to SIP2Net losses
        ssim_alpha: Weight of SSIM loss
        gdl_alpha: Weight of Gradient Difference loss
        f1_alpha: Weight of F1 loss
        
    Returns:
        SIP2NetLoss instance
    """
    return SIP2NetLoss(
        ssim_alpha=ssim_alpha,
        gdl_alpha=gdl_alpha,
        f1_alpha=f1_alpha,
        use_mse=use_mse,
        mse_weight=mse_weight
    )



if __name__ == "__main__":
    print("Testing SIP2Net Loss...")
    criterion = create_sip2net_loss(use_mse=True, mse_weight=0.5, ssim_alpha=500, gdl_alpha=1, f1_alpha=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    batch_size, h, w = 2, 16, 16
    
    target = torch.rand(batch_size, 1, h, w, device=device)
    pred = target.clone()  # Perfect prediction
    mask = torch.ones(batch_size, 1, h, w, device=device)
    
    loss, components = criterion(pred, target, mask)
    
    print("\nTest 1: Identical prediction and target")
    print(f"Total Loss: {loss.item():.6f}")
    print(f"Components: {components}")
    
    target = torch.rand(batch_size, 1, h, w, device=device)
    pred = torch.rand(batch_size, 1, h, w, device=device)
    mask = torch.ones(batch_size, 1, h, w, device=device)
    
    loss, components = criterion(pred, target, mask)
    
    print("\nTest 2: Random prediction and target")
    print(f"Total Loss: {loss.item():.6f}")
    print(f"Components: {components}")
    
    target = torch.rand(batch_size, 1, h, w, device=device)
    pred = torch.rand(batch_size, 1, h, w, device=device)
    mask = torch.zeros(batch_size, 1, h, w, device=device)
    mask[:, :, :h//2, :w//2] = 1.0  # Set top-left quadrant to 1
    
    loss, components = criterion(pred, target, mask)
    
    print("\nTest 3: With partial mask")
    print(f"Total Loss: {loss.item():.6f}")
    print(f"Components: {components}")
    
    target = torch.zeros(batch_size, 1, h, w, device=device)
    pred = torch.zeros(batch_size, 1, h, w, device=device)
    
    for i in range(w):
        target[:, :, :, i] = i / w
        pred[:, :, :, i] = (i / w) * 1.1  # 10% different
    
    mask = torch.ones(batch_size, 1, h, w, device=device)
    
    loss, components = criterion(pred, target, mask)
    
    print("\nTest 4: With gradient information")
    print(f"Total Loss: {loss.item():.6f}")
    print(f"Components: {components}")
    
    print("\nAll tests passed without crashing!")
    
    target = torch.rand(batch_size, 1, h, w, device=device)
    pred = target.clone()  # Perfect prediction
    mask = torch.ones(batch_size, 1, h, w, device=device)
    
    loss, components = criterion(pred, target, mask)
    
    if loss.item() < 0.01:
        print("✓ Loss for identical inputs is close to zero, as expected!")
    else:
        print("✗ Loss for identical inputs is not close to zero!")

