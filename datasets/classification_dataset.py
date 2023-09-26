import os
import torch
import pandas as pd
from random import sample
from PIL import Image
from torch.utils.data import Dataset

from utils.constants import *
from datasets.utils import *



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
    

class RSNAImageDataset(BaseDataset):
    def __init__(self, split="train", transform=None, data_pct=0.01, imsize=None) -> None:
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
            img_path, imsize=self.imsize, transform=self.transform) #default imsize None - no resizing involved

        y = float(row["Target"])
        y = torch.tensor([y])

        return {'image': x,
                'label': y}


class NIHChestXRay14(BaseDataset):
    def __init__(self, split="train", transform=None, data_pct=0.01):
        super().__init__(split=split, transform=transform)

        if not os.path.exists(NIH_CHEST_ROOT_DIR):
            raise RuntimeError(f"{NIH_CHEST_ROOT_DIR} does not exist!")
            
        if data_pct == 0.01:
            train_label_data = "train_1.txt"
        if data_pct == 0.1:
            train_label_data = "train_10.txt"
        if data_pct == 1:
            train_label_data = "train_list.txt"
        test_label_data = "test_list.txt"
        val_label_data = "val_list.txt"
        
        self.root = NIH_CHEST_ROOT_DIR
        self.listImagePaths = []
        self.listImageLabels = []
        
        if self.split == "train":
            downloaded_data_label_txt = train_label_data
        
        elif self.split == "valid":
            downloaded_data_label_txt = val_label_data
                 
        elif self.split == "test":
            downloaded_data_label_txt = test_label_data
           
        #---- Open file, get image paths and labels
        
        fileDescriptor = open(os.path.join(self.root, downloaded_data_label_txt), "r")
        
        #---- get into the loop
        line = True
        
        root_tmp = os.path.join(self.root, "all_classes")
        while line:
                
            line = fileDescriptor.readline()
            
            #--- if not empty
            if line:
                lineItems = line.split()
                imagePath = os.path.join(root_tmp, lineItems[0])
                imageLabel = lineItems[1:]
                imageLabel = [int(i) for i in imageLabel]
                
                self.listImagePaths.append(imagePath)
                self.listImageLabels.append(imageLabel)   
            
        fileDescriptor.close()

    def __getitem__(self, index):
        
        imagePath = self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.listImageLabels[index])

        if self.transform != None: imageData = self.transform(imageData)
        
        return {'image': imageData, 
                'label': imageLabel}

    def __len__(self):
        
        return len(self.listImagePaths)
    


