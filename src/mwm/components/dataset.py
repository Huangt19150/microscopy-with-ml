import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from mwm.components.image_processing import (
    normalize_image, 
    get_gt_mask_png, 
    read_image_png, 
    TestTimeTransform
)
from mwm import logger


def _get_transform(image_size, mode, overlap=0.1):
    if mode == "train":
        return A.Compose([
            A.RandomCrop(width=image_size[0], height=image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5)
        ],
        additional_targets={'sdm': 'mask'}
        )
    elif mode == "val":
        return A.Compose([
            A.CenterCrop(width=image_size[0], height=image_size[1])
        ],
        additional_targets={'sdm': 'mask'}
        )
    elif mode == "test":
        return TestTimeTransform(width=image_size[0], height=image_size[1])
    else:
        logger.error(f"Invalid mode: {mode}")
        raise ValueError(f"Invalid mode: {mode}")


# Utils for custom datasets
def make_dataset(dataset_name, image_dir, mask_dir, sdm_dir, image_list, mode, image_size=[256, 256]):

    transform = _get_transform(image_size, mode)

    if dataset_name == "seg_2ch":
        dataset = Seg2ChannelDataset(image_dir, mask_dir, sdm_dir, image_list, transform)
        logger.info(f"Dataset: {dataset_name} successfully processed. ")
        return dataset
    else:
        logger.error(f"Invalid dataset: {dataset_name}")
        raise ValueError(f"Invalid dataset: {dataset_name}")


# Classes for custom datasets
class Seg2ChannelDataset(Dataset):
    def __init__(self, image_dir, mask_dir, sdm_dir, image_list, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.sdm_dir = sdm_dir
        self.image_list = image_list # This is when image_list is pre-selected for train/val/test split
        self.transform = transform

        # For info retrieval where needed (e.g. at evaluation)
        self.this_image_path = ""
        self.this_mask_path = ""

    def __len__(self):
        return len(self.image_list)


    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the image to retrieve
        Returns:
            image (torch.Tensor): Image tensor of shape (C, H, W).
                Each image only returns one "sample" during training and validation,
                but multiple patches during testing:
                (n_patches_h, n_patches_w, C, patch_h, patch_w)
            mask (torch.Tensor): Mask tensor of shape (C, H, W)
            sdm (torch.Tensor): SDM tensor of shape (1, H, W) (optional)
        """
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.image_list[idx]) # Assuming masks have the same name
        self.this_image_path = img_path
        self.this_mask_path = mask_path

        # Read image and mask
        image = read_image_png(img_path)
        mask_raw = read_image_png(mask_path)
        if self.sdm_dir:
            sdm_path = os.path.join(self.sdm_dir, self.image_list[idx].replace(".png", ".npy")) # Assuming sdms have the same name
            sdm = np.load(sdm_path) # load sdm as numpy array
        else:
            sdm = np.zeros_like(mask_raw).astype(np.float32) # dummy sdm

        # Normalize & Convert to tensors
        image = image / 255.0  # when import from preprocessed image dir: /norm_images
        mask = get_gt_mask_png(mask_raw[:,:,0])[:,:,1:] # leave out the 1st channel (empty), [0 1]
        # mask = get_gt_mask_png(mask_raw[:,:,0])[:,:,-1] # test with nuclei channel only
        # mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
        # mask = mask / 255.0  # Normalize (Assuming mask values are 0 or 255)

        if self.transform:
            augmented = self.transform(image=image, mask=mask, sdm=sdm)
            image = augmented["image"]
            mask = augmented["mask"]
            sdm = augmented["sdm"]

        if len(image.shape) == 5: # Test time
            image = torch.tensor(image, dtype=torch.float32).permute(0, 1, -1, -3, -2)
        else:
            image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        if mask: # Test time doesn't have mask
            mask = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1)
        if sdm: # sdm is optional, depending on the loss in use
            sdm = torch.tensor(sdm, dtype=torch.float32).unsqueeze(0)

        return image, mask, sdm
    
    def get_mask_path(self):
        return self.this_mask_path
    