import torch
import torch.nn.functional as F


class WeightedDiceBCELoss(torch.nn.Module):
    def __init__(self, weight_1=4.0, weight_2=333.3, weight_3=1.0, bce_weight=1.0, epsilon=1e-6):
        """
        Args:
            weight_1: Weight for object foreground in Dice loss.
            weight_2: Weight for boundary foreground in Dice loss.
            weight_3: Weight for boundary channel.
            bce_weight: Weight for binary cross-entropy loss.
            epsilon: Small constant to prevent division by zero.
        """
        super(WeightedDiceBCELoss, self).__init__()
        self.weight_object_foreground = weight_1
        self.weight_boundary_foreground = weight_2
        self.weight_boundary_channel = weight_3
        self.bce_weight = bce_weight
        self.epsilon = epsilon

    def forward(self, logits, targets):
        """
        Args:
            logits: model outputs (assuming already sigmoid), shape (batch_size, 2, H, W)
            targets: Ground truth binary masks (0 or 1), shape (batch_size, 2, H, W)
        """
        # (Apply sigmoid activation)
        # NOTE: Careful not to do it twice or it will demage gradients
        probs = logits # torch.sigmoid(logits)

        # TODO: determine channel identity first. Keep the order consistent with pre/post-processing

        # Apply class weights (higher weight for class 1)
        weights_obj = torch.where(targets[:,1,:,:] == 1, self.weight_object_foreground, 1.0)
        weights_bnd = torch.where(targets[:,0,:,:] == 1, self.weight_boundary_foreground, 1.0)

        # Compute Dice loss for each channel
        boundary_channel_dice = self.dice_loss(
            probs[:,0,:,:], 
            targets[:,0,:,:], 
            weights_bnd, 
            self.epsilon
        )
        object_channel_dice = self.dice_loss(
            probs[:,1,:,:], 
            targets[:,1,:,:], 
            weights_obj, 
            self.epsilon
        )

        # Compute Binary Crossentropy loss for each channel
        boundary_bce = F.binary_cross_entropy(probs[:,0,:,:], targets[:,0,:,:], weights_bnd)
        object_bce = F.binary_cross_entropy(probs[:,1,:,:], targets[:,1,:,:], weights_obj)

        # Combine losses
        total_loss = (self.weight_boundary_channel * boundary_channel_dice + object_channel_dice) + \
                     self.bce_weight * (self.weight_boundary_channel * boundary_bce + object_bce)
    
        return total_loss


    @staticmethod
    def dice_loss(probs, targets, weights=1., epsilon=1e-6):

        # Flatten tensors
        probs = probs.reshape(-1)
        targets = targets.reshape(-1)
        weights = weights.reshape(-1)

        # Compute Dice score
        intersection = torch.sum(weights * probs * targets)
        denominator = torch.sum(weights * probs) + torch.sum(weights * targets)
        
        dice_score = (2. * intersection + epsilon) / (denominator + epsilon)

        return 1. - dice_score