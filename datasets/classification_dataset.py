import os
import torch
import cv2
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from ..constants import *
from .utils import *


class BaseImageDataset(Dataset):
    def __init__(self, split="train", transform=None) -> None:
        super().__init__()

        self.split = split
        self.transform = transform

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    

class RSNAImageDataset(BaseImageDataset):
    def __init__(self, split="train", transform=None,
                 phase="classification", data_pct=0.01, imsize=256) -> None:
        super().__init__(split=split, transform=transform)

        if not os.path.exists(PNEUMONIA_ROOT_DIR):
            raise RuntimeError(f"{PNEUMONIA_ROOT_DIR} does not exist!")

        if self.split == "train":
            self.df = pd.read_csv(PNEUMONIA_TRAIN_CSV)
        elif self.split == "valid":
            self.df = pd.read_csv(PNEUMONIA_VALID_CSV)
        elif self.split == "test":
            self.df = pd.read_csv(PNEUMONIA_TEST_CSV)
        else:
            raise ValueError(f"split {split} does not exist!")

        if phase == "detection":
            self.df = self.df[self.df["Target"] == 1]

        self.df["Path"] = self.df["patientId"].apply(
            lambda x: PNEUMONIA_IMG_DIR / (x + ".dcm"))

        if data_pct != 1 and self.split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)

        self.imsize = imsize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # get image
        img_path = row["Path"]
        x = read_from_dicom(
            img_path, self.imsize, self.transform)
        y = float(row["Target"])
        y = torch.tensor([y])

        return x, y