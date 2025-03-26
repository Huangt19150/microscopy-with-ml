import os
import cv2
import numpy as np
import math
from skimage import measure
from skimage.morphology import dilation, footprint_rectangle, thin
from skimage.segmentation import watershed
from skimage.feature import canny
from scipy import ndimage as ndi
from patchify import patchify


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

    # object_mask_sub_splits = (label > 0).astype(float) * (split_contours==0) # GT mask v1
    object_mask_full = (label > 0).astype(float) # GT mask v2

    # NOTE: room for adding a 3rd channel - first channel removed in training class
    empty_channel = np.zeros_like(label)

    # NOTE: If putting the object_mask_full first help taking the most advantage of pretrained weights?
    #  - No. The 2 channels has no distinction from the pretrained model's perspective: result is pretty random.
    png = np.stack([empty_channel, split_contours, object_mask_full], axis=-1)

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


def post_processing_watershed_2ch(prediction):
    """
    Args:
        - prediction: 2 channels, after thresholding (0/1 values), uint8
    Output:
        - segmentation: 1 channel, labelled objects
    """

    full_foreground = prediction[:,:,1]
    splitlines = prediction[:,:,0]

    # subtracted = np.maximum(full_foreground - splitlines, 0) # This is not working as expected
    subtracted = (full_foreground > 0).astype(float) * (splitlines==0)
    marker = measure.label(subtracted, background=0)
    segmentation = watershed(
        -subtracted,
        marker,
        mask=full_foreground
    )
    return segmentation


def post_processing_denoise_2ch(prediction, dilation_size=3, erosion_size=3):
    """
    Args:
        - prediction: <class 'numpy.ndarray'>, probabilities (float32), 2 channels
        - erosion_size: <int>, size of the erosion kernel
    Output:
        - prediction: <class 'numpy.ndarray'>, 0 or 1 (uint8), 2 channels
    """
    prediction = prediction > 0.5 # boolean

    # fill holes in ch1 (nuclei, full_foreground)
    # (confirmed with IXMtest_N11_s4_w142A84EA3-47C3-4B49-B6CA-BBC6685BBE1E)
    prediction[:,:,1] = ndi.binary_fill_holes(prediction[:,:,1])

    # # erode ch0 (splitlines): not helpful
    # kernel = np.ones((erosion_size, erosion_size), np.uint8)
    # prediction[:,:,0] = cv2.erode(prediction[:,:,0].astype(np.uint8), kernel, iterations=1) # still boolean 

    # # thin ch0 (splitlines) + dilation: not helpful
    # prediction[:,:,0] = thin(prediction[:,:,0].astype(np.uint8))
    # prediction[:,:,0] = dilation(prediction[:,:,0], footprint_rectangle((dilation_size, dilation_size)))

    return prediction.astype(np.uint8)


def pad_image_for_patching(image, patch_size, step):
    """
    Pads an image so it can be fully covered by patches, compatible with step (overlap).
    
    Args:
        image (np.ndarray): Input image, shape (H, W, C)
        patch_size (tuple): Patch size (h, w, c)
        step (int): Step size (stride) between patches.
            e.g. 10% overlap -> step = patch_size * 0.9

    Returns:
        padded_image (np.ndarray): Padded image (target_h, target_w, C)
    """
    h, w, c = image.shape
    patch_h, patch_w, _ = patch_size

    # Calculate required padded dimensions
    n_steps_h = math.ceil((h - patch_h) / step) + 1
    n_steps_w = math.ceil((w - patch_w) / step) + 1

    target_h = step * (n_steps_h - 1) + patch_h
    target_w = step * (n_steps_w - 1) + patch_w

    pad_height = target_h - h
    pad_width = target_w - w

    # Apply padding
    padded_image = np.pad(
        image,
        ((0, pad_height), (0, pad_width), (0, 0)),
        mode='constant',
        constant_values=0
    )

    return padded_image


class TestTimeTransform:
    """
    Test-time augmentation for segmentation models.
    """
    
    def __init__(self, width, height, overlap=0.1):
        self.patch_size = (height, width, 3)
        self.step = int(width * (1 - overlap))

    def __call__(self, image, mask=None, sdm=None):
        """
        Apply test-time augmentation to the input image.
        Args:
            - image (np.ndarray): Input image, shape (H, W, C)
            - mask (np.ndarray): to match up with transforms from albumentations,
                but not used at test time
            - sdm (np.ndarray): to match up with transforms from albumentations,
                but not used at test
        Returns:
            - patches (np.ndarray): Patches of the input image in shape:
                (n_patches_h, n_patches_w, patch_h, patch_w, C)
        """

        # Patchify image
        padded_image = pad_image_for_patching(image, self.patch_size, self.step)
        patches = patchify(padded_image, self.patch_size, self.step)

        # Prepare for full-frame reconstruction
        self.reconstructed = np.zeros_like(padded_image).astype(np.float32)[:,:,:-1] # prediction is 2 channel
        self.recon_weight_map = np.zeros_like(self.reconstructed)
        self.n_patches_h, self.n_patches_w = patches.shape[0], patches.shape[1]
        self.ori_frame_dim0, self.ori_frame_dim1 = image.shape[0], image.shape[1]

        return {
            'image': patches.squeeze(),
            'mask': None,
            'sdm': None
        }

    def reconstruct_full_frame(self, processed_patches):
        """
        Reconstruct the full-frame image from the processed patches.
        Args:
            - processed_patches (np.ndarray): i.e. model predictions, in shape:
                (n_patches_h * n_patches_w, C, patch_h, patch_w)
                e.g. <class 'numpy.ndarray'>, shape: (9, 2, 256, 256)
        Returns:
            - reconstructed (np.ndarray): Full-frame image, shape (H-padded, W-padded, C-as-mask)
        """
        processed_patches = processed_patches.reshape(
            self.n_patches_h, self.n_patches_w, *processed_patches.shape[1:]
        )
        processed_patches = np.transpose(processed_patches, (0, 1, 3, 4, 2))

        # Recon
        for i in range(self.n_patches_h):
            for j in range(self.n_patches_w):
                y = i * self.step
                x = j * self.step
                self.reconstructed[
                        y:y+self.patch_size[0], 
                        x:x+self.patch_size[1],
                        :
                    ] += processed_patches[i, j]
                self.recon_weight_map[
                        y:y+self.patch_size[0], 
                        x:x+self.patch_size[1], 
                        :
                    ] += 1  # Increment weight mask

        self.recon_weight_map[self.recon_weight_map == 0] = 1 # avoid division by zero (just in case)
        self.reconstructed /= self.recon_weight_map.astype(np.float32) # average overlapping regions
        self.reconstructed = self.reconstructed[:self.ori_frame_dim0, :self.ori_frame_dim1, :] # crop to original size

        return self.reconstructed
