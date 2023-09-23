import os
import pickle
import pandas as pd
import numpy as np
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
    

class RSNASegmentDataset(BaseDataset):
    def __init__(self, split="train", transform=None, data_pct=1., imsize=224) -> None:
        super().__init__(split, transform)

        if not os.path.exists(PNEUMONIA_ROOT_DIR):
            raise RuntimeError(f"{PNEUMONIA_ROOT_DIR} does not exist!")

        if self.split == "train":
            with open(PNEUMONIA_DETECTION_TRAIN_PKL, "rb") as f:
                self.filenames, self.bboxs = pickle.load(f)
        elif self.split == "valid":
            with open(PNEUMONIA_DETECTION_VALID_PKL, "rb") as f:
                self.filenames, self.bboxs = pickle.load(f)
        elif self.split == "test":
            with open(PNEUMONIA_DETECTION_TEST_PKL, "rb") as f:
                self.filenames, self.bboxs = pickle.load(f)
        else:
            raise ValueError(f"split {split} does not exist!")

        # self.df["Path"] = self.df["patientId"].apply(
        #     lambda x: RSNA_IMG_DIR / (x + ".dcm"))

        self.imsize = imsize

        n = len(self.filenames)
        if split == "train":
            indices = np.random.choice(n, int(data_pct * n), replace=False)
            self.filenames = self.filenames[indices]
            self.bboxs = self.bboxs[indices]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        img_path = PNEUMONIA_IMG_DIR / filename
        img = read_from_dicom(img_path, imsize=self.imsize)
        x = np.asarray(img)

        mask = np.zeros([1024, 1024])

        bbox = self.bboxs[index]
        new_bbox = bbox[bbox[:, 3] > 0].astype(np.int64)
        if len(new_bbox) > 0:
            for i in range(len(new_bbox)):
                mask[new_bbox[i, 1]:new_bbox[i, 3],
                     new_bbox[i, 0]:new_bbox[i, 2]] += 1
        mask = (mask >= 1).astype("float32")
        mask = resize_img(mask, self.imsize)
        
        augmented = self.transform(image=x, mask=mask)
        x = augmented["image"]
        y = augmented["mask"].squeeze()

        return {"image": x, "mask": y}
