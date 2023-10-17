from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image

from utils.constants import *



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


class VQASLAKEDataset(BaseDataset): # Adapted from M2I2
    def __init__(self, split="train", transform=None):
        super().__init__(split=split, transform=transform) 
        
        if not os.path.exists(SLAKE_DATA_DIR):
            raise RuntimeError(f"{SLAKE_DATA_DIR} does not exist!")
        
        # read in csv file
        if self.split == "train":
            self.df = pd.read_csv(SLAKE_TRAIN_CSV)
        elif self.split == "valid":
            self.df = pd.read_csv(SLAKE_VALID_CSV)
        elif self.split == "test":
            self.df = pd.read_csv(SLAKE_TEST_CSV)
        else:
            raise ValueError(f"split {split} does not exist!")

    def __getitem__(self, index):
        row = self.df.iloc[index]

        # read img
        image_path = row['img_path']
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        question = row["question"]
        question_id = row["qid"]
        answer = row["answer"]

        return {'image': image,
                'question_id': question_id,
                'question': question,
                'answer': answer}

    def __len__(self):
        return len(self.df)