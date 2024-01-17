import os

import lightning as L
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


class CaravanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        pass

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
        # Set channels dimension as second dimension
        image = np.transpose(image, (2, 0, 1))
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
            mask[mask == 255.0] = 1.0
        
        if self.transform is not None:
            image = self.transform(image)
            if self.mask_dir is not None:
                mask = self.transform(mask) 

        return image, mask
    
class CarvanaDatamodule(L.LightningDataModule):
    def __init__(self, data_dir, image_height, image_width, batch_size, num_workers):
        super(CarvanaDatamodule, self).__init__()
        self.data_dir = data_dir
        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size
        self.num_workers = num_workers

        train_transform = v2.Compose([
            v2.Resize((self.image_height, self.image_width), antialias=True),
            v2.RandomRotation(degrees=35),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(p=0.2),
            v2.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0]
            ),
            v2.ToDtype(torch.float32, scale=True),
        ])

        val_transform = v2.Compose([
            v2.Resize((self.image_height, self.image_width), antialias=True),
            v2.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0]
            ),
            v2.ToDtype(torch.float32, scale=True),
        ])

        test_transform = v2.Compose([
            v2.Resize((self.image_height, self.image_width), antialias=True),
            v2.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0]
            ),
            v2.ToDtype(torch.float32, scale=True)
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

        