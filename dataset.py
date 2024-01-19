import os

import lightning as L
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
from torchvision.io import read_image
import random
import torchvision.transforms.v2.functional as F


class CaravanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, train:bool):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.train = train
        self.images = os.listdir(image_dir)
        pass

    def __len__(self):
        return len(self.images)
    
    def _transform(self, image, mask, train):
        '''Transformations are performed through v2 API in
        order to ensure mask and image are transformed in
        the same way.'''

        # Resize
        resize = v2.Resize(size=(160, 240), antialias=True)
        image = resize(image)
        mask = resize(mask)

        # Normalize image
        image = (image - image.mean()) / (image.std() + 1e-8)

        if train:
            # Random rotation
            angle = np.random.randint(-35, 35)
            image = F.rotate(image, angle)
            mask = F.rotate(mask, angle)

            # Random horizontal flip
            if random.random() > 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)
            
            # Random vertical flip
            if random.random() > 0.5:
                image = F.vflip(image)
                mask = F.vflip(mask)

        mask = (mask > 0.5).to(torch.float32) # Binarize the mask

        # Transform to torch tensor
        image = F.to_dtype(image, torch.float32)
        mask = F.to_dtype(mask, torch.float32)

        return image, mask
    
    def __getitem__(self, index):
        # Load image
        img_path = os.path.join(self.image_dir, self.images[index])
        image = read_image(img_path).to(torch.float32)

        # Load mask
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))
            mask = torch.tensor(np.array(Image.open(mask_path), dtype=np.float32), dtype=torch.float32).unsqueeze(0)
            mask[mask == 255.0] = 1.0
        else:
            # If there is no mask this will be ignored
            mask = torch.zeros_like(image)

        # Apply transformations
        image, mask = self._transform(image, mask, self.train)

        return image, mask
    
class CarvanaDatamodule(L.LightningDataModule):
    def __init__(self, data_dir, image_height, image_width, batch_size, num_workers):
        super(CarvanaDatamodule, self).__init__()
        self.data_dir = data_dir
        self.image_height = image_height
        self.image_width = image_width
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Create datasets
        self.train_data = CaravanaDataset(os.path.join(self.data_dir, "train"),
                                          os.path.join(self.data_dir, "train_masks"),
                                          train=True)
        self.val_data = CaravanaDataset(os.path.join(self.data_dir, "validation"),
                                        os.path.join(self.data_dir, "validation_masks"),
                                        train=False)
        self.test_data = CaravanaDataset(os.path.join(self.data_dir, "test"),
                                        None,
                                        train=False)
        
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

        