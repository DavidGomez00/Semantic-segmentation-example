import os

import lightning as L
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class CaravanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"))
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            mask[mask == 255.0] = 1.0
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            if self.mask_dir is not None:
                mask = augmentations["mask"]

        return image, mask
    
class CarvanaDatamodule(L.LightningDataModule):
    def __init__(self, data_dir, image_height, image_width, batch_size, num_workers):
        super(CarvanaDatamodule, self).__init__()
        self.data_dir = data_dir
        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size
        self.num_workers = num_workers

        train_transform = transforms.Compose([
            transforms.Resize(height=self.image_height, width=self.image_width),
            transforms.Rotate(limit=35, p=1.0),
            transforms.HorizontalFlip(p=0.5),
            transforms.VerticalFlip(p=0.1),
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            transforms.ToTensor(),
        ])

        val_transform = transforms.Compose([
            transforms.Resize(height=self.image_height, width=self.image_width),
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(height=self.image_height, width=self.image_width),
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0
            ),
            transforms.ToTensor()
        ])

        self.train_data = CaravanaDataset(os.path.join(self.data_dir, "train"),
                                          os.path.join(self.data_dir, "train_masks"),
                                          transform=train_transform)
        self.val_data = CaravanaDataset(os.path.join(self.data_dir, "validation"),
                                        os.path.join(self.data_dir, "validation_masks"),
                                        transform=val_transform)
        self.test_data = CaravanaDataset(os.path.join(self.data_dir, "test"),
                                        None,
                                        transform=test_transform)
        
    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

        