import torch
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedDiceBCELoss(nn.Module):
    def __init__(self, 
                 weight_1=4.0, 
                 weight_2=333.3, 
                 weight_3=1.0, 
                 bce_weight=1.0, 
                 use_focal=False,
                 use_gradient_loss=False,
                 use_dist_loss=False,
                 focal_alpha=0.25,
                 focal_gamma=2.0,
                 grad_weight=1.0,
                 boundary_dist_weight=1.0,
                 epsilon=1e-6,
    ):
        """
        Args:
            weight_1: Weight for object foreground in Dice loss.
            weight_2: Weight for boundary foreground in Dice loss.
            weight_3: Weight for boundary channel weight.
            bce_weight: Weight for BCE loss.
            epsilon: Small constant to prevent division by zero.
            use_focal: Whether to use Focal Loss for boundary channel.
            focal_gamma: Focal Loss gamma parameter.
            focal_alpha: Focal Loss alpha parameter.
            use_gradient_loss: Whether to use Sobel gradient loss.
            grad_weight: Weight for gradient loss term.
        """
        super(WeightedDiceBCELoss, self).__init__()
        self.weight_object_foreground = weight_1
        self.weight_boundary_foreground = weight_2
        self.weight_boundary_channel = weight_3
        self.bce_weight = bce_weight
        self.epsilon = epsilon

        # Focal Loss params
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        # Gradient loss params
        self.use_gradient_loss = use_gradient_loss
        self.grad_weight = grad_weight

        # Boundary distance weight
        self.use_dist_loss = use_dist_loss
        self.boundary_dist_weight = boundary_dist_weight
        
        # Sobel filter for gradient loss
        if use_gradient_loss:
            sobel_kernel = torch.tensor([
                [[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]],  # x
                [[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]   # y
            ], dtype=torch.float32)
            self.sobel_filter = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)
            self.sobel_filter.weight.data = sobel_kernel
            self.sobel_filter.requires_grad_(False)

    def forward(self, logits, targets, sdm_tensor):
        """
        Args:
            logits: model outputs (assuming sigmoid already applied), shape (B, 2, H, W)
            targets: Ground truth masks, shape (B, 2, H, W)
        """
        probs = logits  # already sigmoid applied, as per note

        # Per-pixel weights
        weights_obj = torch.where(targets[:,1,:,:] == 1, self.weight_object_foreground, 1.0)
        weights_bnd = torch.where(targets[:,0,:,:] == 1, self.weight_boundary_foreground, 1.0)

        # ----- Dice Loss -----
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

        # ----- BCE or Focal Loss -----
        if self.use_focal:
            boundary_bce = self.focal_loss(
                probs[:,0,:,:], targets[:,0,:,:], weights_bnd,
                gamma=self.focal_gamma, alpha=self.focal_alpha
            )
        else:
            boundary_bce = F.binary_cross_entropy(probs[:,0,:,:], targets[:,0,:,:], weights_bnd)

        object_bce = F.binary_cross_entropy(probs[:,1,:,:], targets[:,1,:,:], weights_obj)

        # ----- Gradient Loss -----
        grad_loss = 0.0
        if self.use_gradient_loss:
            grad_loss = self.gradient_loss(probs[:,0,:,:], targets[:,0,:,:])

        # ----- Boundary Distance Loss -----
        boundary_dist_loss = 0.0
        if self.use_dist_loss:
            boundary_dist_loss = torch.mean(probs[:,0,:,:] * sdm_tensor.to(probs.device))

        # ----- Total Loss -----
        total_loss = (self.weight_boundary_channel * boundary_channel_dice + object_channel_dice) + \
                     self.bce_weight * (self.weight_boundary_channel * boundary_bce + object_bce) + \
                     self.grad_weight * grad_loss + \
                     self.boundary_dist_weight * boundary_dist_loss

        return total_loss

    @staticmethod
    def dice_loss(probs, targets, weights=1., epsilon=1e-6):
        probs = probs.reshape(-1)
        targets = targets.reshape(-1)
        weights = weights.reshape(-1)

        intersection = torch.sum(weights * probs * targets)
        denominator = torch.sum(weights * probs) + torch.sum(weights * targets)
        dice_score = (2. * intersection + epsilon) / (denominator + epsilon)
        return 1. - dice_score

    @staticmethod
    def focal_loss(probs, targets, weights=1., gamma=2.0, alpha=0.25):
        probs = probs.reshape(-1)
        targets = targets.reshape(-1)
        weights = weights.reshape(-1)

        bce = F.binary_cross_entropy(probs, targets, reduction='none')
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_term = (1 - p_t) ** gamma
        loss = alpha * focal_term * bce * weights
        return loss.mean()

    def gradient_loss(self, pred, target):
        pred = pred.unsqueeze(1)
        target = target.unsqueeze(1)

        # Move sobel filter to the same device as input
        self.sobel_filter = self.sobel_filter.to(pred.device)
        
        pred_grad = self.sobel_filter(pred)
        target_grad = self.sobel_filter(target)
        return F.l1_loss(pred_grad, target_grad)
