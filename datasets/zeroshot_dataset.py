import os
import torch
import cv2
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset

import sys
from pathlib import Path
path_root = Path(__file__).parents[2]
from constants import *
from datasets.utils import *



class MIMIC_5x200(Dataset): 
    def __init__(self, transform=None, tokenizer=None) -> None:
        super().__init__()

        self.transform = transform(is_train=False)

        if not os.path.exists(MIMIC_CXR_ROOT_DIR):
            raise RuntimeError(f"{MIMIC_CXR_ROOT_DIR} does not exist!")
        elif not os.path.exists(MIMIC_CXR_5X200):
            raise RuntimeError("Please pre-process MIMIC-CXR-5x200 dataset")
        
        self.df = pd.read_csv(MIMIC_CXR_5X200)
        self.df[MIMIC_CXR_REPORT_COL] = self.df[['findings', 'impression']].agg(' '.join, axis=1)
        self.df[MIMIC_CXR_PATH_COL] = self.df[MIMIC_CXR_PATH_COL].apply(lambda x: MIMIC_CXR_ROOT_DIR / x)

        self.listImagePaths = self.df[MIMIC_CXR_PATH_COL].tolist()
        self.listReports = self.df[MIMIC_CXR_REPORT_COL].tolist()
        self.tokenizer = tokenizer
    
    def process_report(self, report, max_length=97):
        text_inputs = self.tokenizer(report, truncation=True, padding="max_length", 
                                     return_tensors='pt', max_length=max_length)
        return text_inputs

    def process_img(self, path):
        x = cv2.imread(str(path), 0)

        # transform images
        img = Image.fromarray(x).convert("RGB")
        img = self.transform(img)

        return img
    
    def __getitem__(self, index):

        imagePath = self.listImagePaths[index]
        report = self.listReports[index]

        processed_img = self.process_img(imagePath)
        processed_txt =self.process_report(report)

        target = torch.tensor(self.df[CHEXPERT_COMPETITION_TASKS].values[index])

        return processed_img, processed_txt, target
    
    def __len__(self):
        return len(self.df)
    
    def collate_fn(self, batches):

        image_list, text_list, target_list = [], [], []

        for sample in batches:
            img, text, target = sample
            image_list.append(img)
            text_list.append(text)
            target_list.append(target)

        input_ids = torch.stack([x["input_ids"] for x in text_list]).squeeze()
        attention_masks = torch.stack([x["attention_mask"] for x in text_list]).squeeze()
        token_type_ids = torch.stack([x["token_type_ids"] for x in text_list]).squeeze()
        
        all_imgs = torch.stack(image_list)
        all_txt = {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "token_type_ids": token_type_ids,
        }
        all_targets = torch.stack(target_list)    

        return {"image": all_imgs,
                "text": all_txt,
                "target": all_targets}

        
# Zero-shot image-text retrieval
class CheXpert_5x200(Dataset): 
    def __init__(self, transform=None, tokenizer=None) -> None:
        super().__init__()

        self.transform = transform(is_train=False)

        if not os.path.exists(CHEXPERT_DATA_DIR):
            raise RuntimeError(f"{CHEXPERT_DATA_DIR} does not exist!")
        elif not os.path.exists(CHEXPERT_DATA_DIR):
            raise RuntimeError("Please pre-process MIMIC-CXR-5x200 dataset")
        
        self.df = pd.read_csv(CHEXPERT_5x200)
        self.df[CHEXPERT_REPORT_COL] = self.df[['findings', 'impression']].agg(' '.join, axis=1)

        self.listImagePaths = self.df[CHEXPERT_PATH_COL].tolist()
        self.listReports = self.df[CHEXPERT_PATH_COL].tolist()
        self.tokenizer = tokenizer
    
    def process_report(self, report, max_length=97):
        text_inputs = self.tokenizer(report, truncation=True, padding="max_length", 
                                     return_tensors='pt', max_length=max_length)
        return text_inputs

    def process_img(self, path):
        x = cv2.imread(str(path), 0)

        # transform images
        img = Image.fromarray(x).convert("RGB")
        img = self.transform(img)

        return img
    
    def __getitem__(self, index):

        imagePath = self.listImagePaths[index]
        report = self.listReports[index]

        processed_img = self.process_img(imagePath)
        processed_txt =self.process_report(report)

        target = torch.tensor(self.df[CHEXPERT_COMPETITION_TASKS].values[index])

        return processed_img, processed_txt, target
    
    def __len__(self):
        return len(self.df)
    
    def collate_fn(self, batches):

        image_list, text_list, target_list = [], [], []

        for sample in batches:
            img, text, target = sample
            image_list.append(img)
            text_list.append(text)
            target_list.append(target)

        input_ids = torch.stack([x["input_ids"] for x in text_list]).squeeze()
        attention_masks = torch.stack([x["attention_mask"] for x in text_list]).squeeze()
        token_type_ids = torch.stack([x["token_type_ids"] for x in text_list]).squeeze()
        
        all_imgs = torch.stack(image_list)
        all_txt = {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "token_type_ids": token_type_ids,
        }
        all_targets = torch.stack(target_list)    

        return {"image": all_imgs,
                "text": all_txt,
                "target": all_targets}
        
