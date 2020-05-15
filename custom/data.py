""" Custom dataset classes / scripts.
"""

import os
import random
import glob
from PIL import Image
import torch

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, transform=None, num_imgs=None):
        super(MyDataset).__init__()
        self.num_imgs = num_imgs
        self.transform = transform

        # find all files
        self.files = glob.glob(filepath + "*.jpg")
        random.shuffle(self.files)
        
        # limit the number of images (if specified)
        if self.num_imgs is not None:
            self.files = self.files[:self.num_imgs]

    def __getitem__(self, idx):
        # return the file and label of the corresponding index
        animal = os.path.basename(self.files[idx]).split(".")[0]
        label = 0 if animal=="dog" else 1
        image = Image.open(self.files[idx]).convert('L')

        if self.transform is not None:
            image = self.transform(image)

        return (image, label)

    def __len__(self):
        return len(self.files)
