import os
import cv2
import numpy as np
from skimage import measure
from skimage.morphology import dilation, footprint_rectangle
from skimage.feature import canny

import matplotlib.pyplot as plt

def normalize_image(image):
    """
    Args:
        image: 2D array of image
        high_pass_threshold: Set based on general max value intensity of raw image
    """

    return (image - np.min(image)) / (np.max(image) - np.min(image))

def get_contours(mask, dilation_size=2, scale_255=False):

    mask = (normalize_image(mask) * 255.).astype(np.uint8)

    # Use Canny edge detection
    edges = canny(mask)

    # Dilate edges to make them more continuous
    edges_dilated = dilation(edges, footprint_rectangle((dilation_size, dilation_size)))
    contour_mask = edges_dilated.astype(float) # convert bool to 0/1

    if scale_255:
        contour_mask = contour_mask * 255

    return contour_mask

def get_split_contours(label, dilation_size_1=1, dilation_size_2=1, dilation_size_3=3, scale_255=False):
    """
    Args:
        label: 2D array of labelled objects
        dilation_size_1: Size of dilation for object contours
        dilation_size_2: Size of dilation for aggregated object mask: >= dilation_size_1 to reduce false split-lines (noise)
        dilation_size_3: Size of dilation for split contours output: dilation is required to ensure nuclei mask do nut touch anykmore
        scale_255: Whether to scale the output to 0-255
    """

    # Create a mask of all labelled objects aggregated
    mask_aggregated = (label > 0).astype(np.uint8) * 255

    contour_label = get_contours(label, dilation_size_1)
    contour_mask = get_contours(mask_aggregated, dilation_size_2)
    split_contours = np.maximum((contour_label - contour_mask), 0)

    # TODO: Remove noise segments / further dialation?
    split_contours = split_contours * (label > 0) # remove contours outside of objects

    split_contours = dilation(split_contours, footprint_rectangle((dilation_size_3, dilation_size_3)))

    if scale_255:
        split_contours = split_contours * 255

    return split_contours

def get_gt_mask_png(label, dilation_size_1=1, dilation_size_2=1, dilation_size_3=3, save_path=None):

    split_contours = get_split_contours(label, dilation_size_1, dilation_size_2, dilation_size_3)

    object_mask_sub_splits = (label > 0).astype(float) * (split_contours==0)

    # NOTE: might be just for visualization of PNG - remove during training
    empty_channel = np.zeros_like(label)
    png = np.stack([empty_channel, split_contours, object_mask_sub_splits], axis=-1)

    if save_path:
        pass

    return png


def read_image_png(image_path):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
    else:
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"OpenCV could not read: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    return image