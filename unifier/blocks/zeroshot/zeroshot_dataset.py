import os
import torch
import cv2
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset

from .prompts import generate_chexpert_class_prompts, generate_rsna_class_prompts
import unifier.datasets.transforms as transforms


_CSVPATH = {
    "mimic_5x200": "unifier/datasets/data/mimic_5x200.csv.zip",
    "chexpert_5x200": "unifier/datasets/data/chexpert_5x200.csv.zip"
}


CHEXPERT_COMPETITION_TASKS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]


class MIMIC_5x200(Dataset):
    def __init__(self, img_path, transform=None, tokenizer=None) -> None:
        super().__init__()

        self.transform = eval("transforms." + transform)(is_train=False)

        if not os.path.exists(img_path):
            raise RuntimeError(f"{img_path} does not exist!")
        
        self.df = pd.read_csv(_CSVPATH["mimic_5x200"])
        self.df["Path"] = self.df["Path"].apply(lambda x: os.path.join(img_path, x))

        self.listImagePaths = self.df["Path"].tolist()
        self.listReports = self.df["Report"].tolist()
        self.tokenizer = tokenizer
    
    def process_img(self, path):
        x = cv2.imread(str(path), 0)

        # transform images
        img = Image.fromarray(x).convert("RGB")
        img = self.transform(img)

        return img
    
    def process_report(self, report, max_length=97):
        text_inputs = self.tokenizer(report, truncation=True, padding="max_length", 
                                     return_tensors='pt', max_length=max_length)
        
        cap_lens = len([w for w in report if not w.startswith("[")])
        text_inputs["cap_lens"] = torch.tensor(cap_lens)
        
        return text_inputs
    
    def __getitem__(self, index):

        imagePath = self.listImagePaths[index]
        report = self.listReports[index]

        processed_img = self.process_img(imagePath)
        processed_txt = self.process_report(report)

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
        
        all_imgs = torch.stack(image_list)
        all_txt = {k: torch.stack([dic[k] for dic in text_list]).squeeze() for k in text_list[0]}
        all_targets = torch.stack(target_list)    

        return {"image": all_imgs,
                "text": all_txt,
                "target": all_targets}

        
class CheXpert_5x200(Dataset): 
    def __init__(self, img_path, transform=None, tokenizer=None) -> None:
        super().__init__()

        self.transform = eval("transforms." + transform)(is_train=False)

        if not os.path.exists(img_path):
            raise RuntimeError(f"{img_path} does not exist!")
        
        self.df = pd.read_csv(_CSVPATH["chexpert_5x200"])
        self.listImagePaths = self.df["Path"].tolist()

    def process_img(self, path):
        x = cv2.imread(str(path), 0)

        # transform images
        img = Image.fromarray(x).convert("RGB")
        img = self.transform(img)

        return img
    
    def __getitem__(self, index):
        imagePath = self.listImagePaths[index]
        processed_img = self.process_img(imagePath)
        target = torch.tensor(self.df[CHEXPERT_COMPETITION_TASKS].values[index])
        return processed_img, target # No reports
    
    def __len__(self):
        return len(self.df)
    
    def collate_fn(self, batches):
        image_list, target_list = [], []

        for sample in batches:
            img, target = sample
            image_list.append(img)
            target_list.append(target)

        all_imgs = torch.stack(image_list)
        all_targets = torch.stack(target_list)   
 
        return {"image": all_imgs,
                "target": all_targets}