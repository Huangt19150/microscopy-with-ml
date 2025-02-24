import os
import cv2
import torch
from torch.utils.data import Dataset
from mwm.components.image_processing import normalize_image, get_gt_mask_png


class Seg2ChannelDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_list, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_list = image_list # This is when image_list is pre-selected for train/val/test split
        self.transform = transform # TODO:

    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def _read_image_png(image_path):
        if not os.path.exists(image_path):
            print(f"File not found: {image_path}")
        else:
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"OpenCV could not read: {image_path}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        return image

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        mask_path = os.path.join(self.mask_dir, self.image_list[idx])  # Assuming masks have the same name

        # Read image and mask
        image = self._read_image_png(img_path)
        mask_raw = self._read_image_png(mask_path)

        # Normalize & Convert to tensors
        image = image / 255.0  # when import from preprocessed image dir: /norm_images
        mask = get_gt_mask_png(mask_raw[:,:,0])[:,:,1:] # leave out the 1st channel (empty), [0 1]
        # mask = get_gt_mask_png(mask_raw[:,:,0])[:,:,-1] # test with nuclei channel only
        # mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
        # mask = mask / 255.0  # Normalize (Assuming mask values are 0 or 255)

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        mask = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1)

        return image, mask
    