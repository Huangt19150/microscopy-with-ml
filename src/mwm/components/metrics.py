import torch

def iou_base(preds, masks, threshold=0.5, eps=1e-6):
    """
    IoU by definition regardless of shape.

    Args:
        - preds: Predictions from the model
        - masks: Ground truth masks
        - threshold: Threshold for binarization
        - eps: Small constant to prevent division by zero
    Output:
        - (Float): Intersection over Union (IoU) score
    """
    # Flatten everything
    preds = preds.reshape(-1)
    masks = masks.reshape(-1)

    preds = (preds > threshold).float()
    intersection = torch.sum(preds * masks)
    union = torch.sum(preds) + torch.sum(masks) - intersection
    return (intersection + eps) / (union + eps)


def iou_list(preds, masks, threshold=0.5, eps=1e-6):
    """
    IoU image-wise when preds and masks are batched.

    Args:
        - preds: Predictions from the model (B x H x W)
        - masks: Ground truth masks (B x H x W)
        - threshold: Threshold for binarization
        - eps: Small constant to prevent division by zero
    Output:
        - (List): List of IoU scores for each image
    """
    iou_list = []
    for i in range(preds.shape[0]):
        iou = iou_base(preds[i,:,:], masks[i,:,:], threshold, eps)
        iou_list.append(iou)
    return iou_list