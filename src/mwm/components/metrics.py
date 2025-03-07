import numpy as np
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


def iou_object_labels(ground_truth, prediction):
    """
    Compute the pixel-level segmentation IoU for multiple objects in one image sample.
    Args:
        - ground_truth: 2D numpy array (int), ground truth mask, each object identified by pixels with the same unique value
        - prediction: 2D numpy array (int), predicted mask, each object identified by pixels with the same unique value
    Returns:
        - iou_matrix: 2D numpy array (float), IoU matrix for each object pair
    """
    
    # Count objects
    true_objects = len(np.unique(ground_truth))
    pred_objects = len(np.unique(prediction))
    
    # Compute intersection
    h = np.histogram2d(ground_truth.flatten(), prediction.flatten(), bins=(true_objects,pred_objects))
    intersection = h[0]
    
    # Area of objects
    area_true = np.histogram(ground_truth, bins=true_objects)[0]
    area_pred = np.histogram(prediction, bins=pred_objects)[0]
    
    # Calculate union
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    union = area_true + area_pred - intersection
    
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    
    # Compute Intersection over Union
    union[union == 0] = 1e-9
    iou_matrix = intersection/union
    
    return iou_matrix


def measures_at(threshold, iou_matrix):
    """
    Object level evaluation metrics at a given IoU threshold.
    Args:
        - threshold: float, IoU threshold
        - iou_matrix: 2D numpy array (float), IoU matrix for each object pair
    Returns:
        - f1: float, F1 score
        - precision: float, precision
        - recall: float, recall
        - jaccard: float, Jaccard index
        - TP: int, number of true positives objects identified in a image sample
        - FP: int, number of false positives objects identified in a image sample
        - FN: int, number of false negatives objects identified in a image sample
    """
    
    matches = iou_matrix > threshold
    
    true_positives = np.sum(matches, axis=1) == 1   # Correct objects
    false_positives = np.sum(matches, axis=0) == 0  # Extra objects
    false_negatives = np.sum(matches, axis=1) == 0  # Missed objects
    
    assert np.all(np.less_equal(true_positives, 1))
    assert np.all(np.less_equal(false_positives, 1))
    assert np.all(np.less_equal(false_negatives, 1))
    
    TP, FP, FN = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
    
    f1 = 2*TP / (2*TP + FP + FN + 1e-9)
    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    jaccard = TP / (TP + FP + FN + 1e-9)
    
    return f1, precision, recall, jaccard, TP, FP, FN
