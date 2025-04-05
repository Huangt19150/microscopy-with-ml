import numpy as np


def colorize_label_map(label_map):
    unique_labels = np.unique(label_map)
    n = unique_labels.max() + 1

    # Random colormap: shape (n_labels, 3), dtype=uint8
    np.random.seed(42) # for reproducibility
    colormap = np.random.randint(0, 255, size=(n, 3), dtype=np.uint8, )
    colormap[0] = [0, 0, 0]  # make background black if label 0

    # Create color image
    h, w = label_map.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)

    for label in unique_labels:
        color_image[label_map == label] = colormap[label]

    return color_image


def colorize_mask_2d(mask_prediction):
    empty_channel = np.zeros_like(mask_prediction[:,:,0])
    color_mask = np.stack([mask_prediction[:,:,1], empty_channel, mask_prediction[:,:,0]], axis=-1) * 255

    return color_mask
