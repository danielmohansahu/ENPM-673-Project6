""" Custom dataset classes / scripts.
"""

import os
import glob
from PIL import Image
import torch

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, filepath, train, transform=None):
        super(MyDataset).__init__()
        self.train = train
        self.transform = transform

        # find all files
        self.files = glob.glob(filepath + "*.jpg")

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
