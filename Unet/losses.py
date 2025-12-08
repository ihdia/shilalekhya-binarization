import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_coeff(inputs, targets, smooth=1.0):
    """
    Calculates the Dice coefficient for a batch of predictions. This is a "hard" Dice,
    meaning it operates on binarized predictions.

    Args:
        inputs (torch.Tensor): The model's raw logits (B, C, H, W).
        targets (torch.Tensor): The ground truth masks (B, C, H, W), containing 0s and 1s.
        smooth (float): A smoothing factor to prevent division by zero.

    Returns:
        torch.Tensor: The Dice coefficient score.
    """
    # Apply sigmoid to convert logits to probabilities
    inputs_prob = torch.sigmoid(inputs)

    # Binarize the predictions using a 0.5 threshold
    inputs_binary = (inputs_prob > 0.5).float()

    # Flatten the tensors to compute the coefficient over the whole batch
    inputs_flat = inputs_binary.view(-1)
    targets_flat = targets.view(-1)

    # Calculate the intersection and the sums of the masks
    intersection = (inputs_flat * targets_flat).sum()
    dice_score = (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)

    return dice_score


def iou_coeff(inputs, targets, smooth=1.0):
    """
    Calculates the Intersection over Union (IoU) coefficient, also known as the Jaccard index.
    This is a "hard" IoU, meaning it operates on binarized predictions.

    Args:
        inputs (torch.Tensor): The model's raw logits (B, C, H, W).
        targets (torch.Tensor): The ground truth masks (B, C, H, W), containing 0s and 1s.
        smooth (float): A smoothing factor to prevent division by zero.

    Returns:
        torch.Tensor: The IoU coefficient score.
    """
    # Apply sigmoid to convert logits to probabilities
    inputs_prob = torch.sigmoid(inputs)

    # Binarize the predictions using a 0.5 threshold
    inputs_binary = (inputs_prob > 0.5).float()

    # Flatten the tensors to compute the coefficient over the whole batch
    inputs_flat = inputs_binary.view(-1)
    targets_flat = targets.view(-1)

    # Calculate intersection and union
    intersection = (inputs_flat * targets_flat).sum()
    total = (inputs_flat + targets_flat).sum()
    union = total - intersection

    # Calculate the IoU score
    iou_score = (intersection + smooth) / (union + smooth)

    return iou_score

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        """
        Forward pass for Dice Loss.
        Args:
            inputs (torch.Tensor): The model's raw logits (B, 1, H, W).
            targets (torch.Tensor): The ground truth masks (B, 1, H, W).
            smooth (int): A smoothing factor to prevent division by zero.
        """
        # Apply sigmoid to get probabilities
        inputs = torch.sigmoid(inputs)
        
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, reduction='mean'):
        """
        Focal Loss, as described in https://arxiv.org/abs/1708.02002.
        It is essentially an enhancement to cross-entropy loss and is
        useful for classification tasks when there is a large class imbalance.
        
        Args:
            alpha (float): Weighting factor for the rare class.
            gamma (int): Focusing parameter to down-weight easy examples.
            reduction (str): 'mean', 'sum' or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: raw logits
        # targets: binary ground truth
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) # Prevents nans when probability is 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

class DiceBCELoss(nn.Module):
    """
    A hybrid loss function combining Dice Loss and Binary Cross-Entropy Loss.
    """
    def __init__(self, weight_dice=0.5, weight_bce=0.5):
        super(DiceBCELoss, self).__init__()
        self.dice_loss = DiceLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce

    def forward(self, inputs, targets, smooth=1):
        dice = self.dice_loss(inputs, targets, smooth)
        bce = self.bce_loss(inputs, targets)
        
        # You can weigh the contribution of each loss
        loss = self.weight_bce * bce + self.weight_dice * dice
        return loss
    
class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) Loss, also known as Jaccard Loss.
    """
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        """
        Forward pass for IoU Loss.
        Args:
            inputs (torch.Tensor): The model's raw logits (B, 1, H, W).
            targets (torch.Tensor): The ground truth masks (B, 1, H, W).
            smooth (int): A smoothing factor to prevent division by zero.
        """
        # Apply sigmoid to get probabilities
        inputs = torch.sigmoid(inputs)
        
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # intersection is executing dot product with broadcast
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
        
        total_loss = F.mse_loss(IoU, torch.tensor(1.0, device=inputs.device))
                
        return total_loss
    
# --- COMPOSITE LOSS ---
class FocalDiceIoUMSELoss(nn.Module):
    """
    A composite loss function that combines Focal Loss, Dice Loss, IoU Loss, and MSE Loss.
    The final loss is a weighted sum of the individual losses.
    """
    def __init__(self, focal_weight=20.0, dice_weight=1.0, iou_weight=1.0, mse_weight=1.0):
        super(FocalDiceIoUMSELoss, self).__init__()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.iou_loss = IoULoss()
        self.mse_loss = nn.MSELoss()
        
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.iou_weight = iou_weight
        self.mse_weight = mse_weight
        print(f'Initialized composite loss with weights: Focal={focal_weight}, Dice={dice_weight}, IoU={iou_weight}, MSE={mse_weight}')


    def forward(self, inputs, targets, smooth=1):
        """
        Calculates the combined loss.
        Args:
            inputs (torch.Tensor): Raw logits from the model.
            targets (torch.Tensor): Ground truth binary masks.
        """
        # Calculate individual losses
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets, smooth)
        iou = self.iou_loss(inputs, targets, smooth)
        
        # Calculate the weighted sum
        total_loss = (self.focal_weight * focal +
                      self.dice_weight * dice +
                      self.iou_weight * iou)
                      
        return total_loss