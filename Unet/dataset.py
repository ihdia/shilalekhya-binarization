import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class BinarizationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        # Assuming mask has the same filename as the image
        mask_path = os.path.join(self.mask_dir, self.images[index])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") # Grayscale

        # Default transformation to tensor and normalize mask
        to_tensor = transforms.ToTensor()
        image = to_tensor(image)
        mask = to_tensor(mask) # This will be [0.0, 1.0]

        # Apply custom transforms if provided (like normalization, augmentation)
        if self.transform:
            image = self.transform(image)
            
        return image, mask