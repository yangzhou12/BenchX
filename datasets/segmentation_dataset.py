import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from .utils import *

import sys
from pathlib import Path

path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from constants import *
from datasets.transforms import SegmentationTransforms


class BaseDataset(Dataset):
    def __init__(self, split="train", transform=None) -> None:
        super().__init__()

        self.split = split

        if split == "train":
            self.transform = transform(is_train=True)
        else:
            self.transform = transform(is_train=False)

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class SIIMImageDataset(BaseDataset):
    def __init__(self, split="train", transform=None, data_pct=0.01, imsize=224):
        super().__init__(split=split, transform=transform)

        if not os.path.exists(PNEUMOTHORAX_DATA_DIR):
            raise RuntimeError(f"{PNEUMOTHORAX_DATA_DIR} does not exist!")

        # read in csv file
        if self.split == "train":
            self.df = pd.read_csv(PNEUMOTHORAX_TRAIN_CSV)
        elif self.split == "valid":
            self.df = pd.read_csv(PNEUMOTHORAX_VALID_CSV)
        elif self.split == "test":
            self.df = pd.read_csv(PNEUMOTHORAX_TEST_CSV)
        else:
            raise ValueError(f"split {split} does not exist!")

        # self.df["Path"] = self.df["Path"].apply(
        #     lambda x: os.path.join(PNEUMOTHORAX_IMG_DIR, x)
        # )

        # only keep positive samples for segmentation
        self.df["class"] = self.df[" EncodedPixels"].apply(lambda x: x != " -1")
        if split == "train":
            self.df_neg = self.df[self.df["class"] == False]
            self.df_pos = self.df[self.df["class"] == True]
            n_pos = self.df_pos["ImageId"].nunique()
            neg_series = self.df_neg["ImageId"].unique()
            neg_series_selected = np.random.choice(
                neg_series, size=n_pos, replace=False
            )
            self.df_neg = self.df_neg[self.df_neg["ImageId"].isin(neg_series_selected)]
            self.df = pd.concat([self.df_pos, self.df_neg])

        if data_pct != 1 and split == "train":
            ids = self.df["ImageId"].unique()
            n_samples = int(len(ids) * data_pct)
            series_selected = np.random.choice(ids, size=n_samples, replace=False)
            self.df = self.df[self.df["ImageId"].isin(series_selected)]

        self.img_ids = self.df.ImageId.unique().tolist()
        self.imsize = imsize

    def __getitem__(self, index):
        imgid = self.img_ids[index]
        imgid_df = self.df.groupby("ImageId").get_group(imgid)

        # get image
        img_path = imgid_df.iloc[0]["Path"]
        img = read_from_dicom(img_path, imsize=self.imsize)
        x = np.asarray(img)

        rle_list = imgid_df[" EncodedPixels"].tolist()
        mask = np.zeros([1024, 1024])
        if rle_list[0] != " -1":
            for rle in rle_list:
                mask += self.rle2mask(rle, PNEUMOTHORAX_IMG_SIZE, PNEUMOTHORAX_IMG_SIZE)
        mask = (mask >= 1).astype("float32")
        mask = resize_img(mask, self.imsize)

        augmented = self.transform(image=x, mask=mask)
        x = augmented["image"]
        y = augmented["mask"].squeeze()

        return {"image": x, "mask": y}

    def __len__(self):
        return len(self.img_ids)

    def rle2mask(self, rle, width, height):
        """Run length encoding to segmentation mask"""

        mask = np.zeros(width * height)
        array = np.asarray([int(x) for x in rle.split()])
        starts = array[0::2]
        lengths = array[1::2]
        current_position = 0
        for index, start in enumerate(starts):
            current_position += start
            mask[current_position : current_position + lengths[index]] = 1
            current_position += lengths[index]

        return mask.reshape(width, height)
