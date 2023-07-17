from copy import deepcopy
import os
from typing import List, Tuple
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
import tokenizers
import random


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class MultimodalBertDataset(Dataset): #Search PyTorch documentation - abstracts preprocessing of data
    def __init__(
        self,
        data_root,
        transform,
        max_caption_length: int = 100
    ):
        self.max_caption_length = max_caption_length
        self.data_root = data_root # directory where data is stored
        self.transform = transform # Transform object
        self.images_list, self.report_list = self.read_csv() # see method below - images array and report array
        self.tokenizer = tokenizers.Tokenizer.from_file("mimic_wordpiece.json")
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def __len__(self):
        return len(self.images_list)
    
    def _random_mask(self,tokens): #applies word-level random masking with 50% probability
        masked_tokens = deepcopy(tokens)
        for i in range(1, masked_tokens.shape[1]-1): #shape[1] - each token tensor length
            # refer to mimic_wordpiece.json for vocabulary

            if masked_tokens[0][i] == 0: #empty string
                break
            
            if masked_tokens[0][i-1] == 3 and self.idxtoword[masked_tokens[0][i].item()][0:2] == '##': #if previous token is masked and current token is part of word, mask current token
                masked_tokens[0][i] = 3
                continue
            
            if masked_tokens[0][i-1] != 3 and self.idxtoword[masked_tokens[0][i].item()][0:2] == '##': #if previous token is unmasked and current token is part of word, don't mask any sub-words
                continue

            prob = random.random()
            if prob < 0.5: # mask each word with 50% probability
                masked_tokens[0][i] = 3

        return masked_tokens

    def __getitem__(self, index): #tokenization logic goes here
        image = pil_loader(self.images_list[index]) #load img
        image = self.transform(image) #transform img according to defined transformations
        sent = self.report_list[index] #sentence/report
        sent = '[CLS] '+ sent #adding special tokens for BERT
        self.tokenizer.enable_truncation(max_length=self.max_caption_length) #truncation/padding for equal size encodings
        self.tokenizer.enable_padding(length=self.max_caption_length)

        encoded = self.tokenizer.encode(sent) #encode report
        ids = torch.tensor(encoded.ids).unsqueeze(0)
        attention_mask = torch.tensor(encoded.attention_mask).unsqueeze(0) # attention mask - shows where the padding positions are
        type_ids = torch.tensor(encoded.type_ids).unsqueeze(0) # unsqueeze(0): inserts a new dimension of size 1 at dimension 0 - [...] -> [[...]]
        masked_ids = self._random_mask(ids)
        return image, ids, attention_mask, type_ids, masked_ids
    
    def read_csv(self):
        csv_path = os.path.join(self.data_root,'training2.csv')
        df = pd.read_csv(csv_path,sep=',')
        return df["image_path"], df["report_content"]

    def collate_fn(self, instances: List[Tuple]):
        image_list, ids_list, attention_mask_list, type_ids_list, masked_ids_list = [], [], [], [], []
        # flatten
        for b in instances:
            image, ids, attention_mask, type_ids, masked_ids = b
            image_list.append(image)
            ids_list.append(ids)
            attention_mask_list.append(attention_mask)
            type_ids_list.append(type_ids)
            masked_ids_list.append(masked_ids)

        # stack
        image_stack = torch.stack(image_list) #torch.stack - concatenates a sequence of tensors along a new dimension
        ids_stack = torch.stack(ids_list).squeeze() # [id_tensor] -> [[id_tensor]]
        attention_mask_stack = torch.stack(attention_mask_list).squeeze()
        type_ids_stack = torch.stack(type_ids_list).squeeze()
        masked_ids_stack = torch.stack(masked_ids_list).squeeze()

        # sort and add to dictionary
        # returned batch object
        return_dict = {
            "image": image_stack,
            "labels": ids_stack,
            "attention_mask": attention_mask_stack,
            "type_ids": type_ids_stack,
            "ids": masked_ids_stack
        }

        return return_dict