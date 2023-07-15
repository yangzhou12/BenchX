import os
import pickle
import re
import torch
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
from PIL import Image
from ..constants import *
from .utils import *


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class MultimodalPretrainingDataset(Dataset):
    def __init__(self, split="train", transform=None, data_pct=1.0, imsize=224):
        super().__init__()
        
        if not os.path.exists(MIMIC_CXR_ROOT_DIR):
            raise RuntimeError(
                "MIMIC-CXR does not exist!\n"
                + "Make sure to download data from:\n"
                + "    https://physionet.org/content/mimic-cxr/2.0.0/"
                + " and update MIMIC_ROOT_DIR in ./constants.py"
            )
        
        self.transform = transform
        self.imsize = imsize
        
        # Load data
        self.df = pd.read_csv(MIMIC_CXR_MASTER_CSV)
        self.df[MIMIC_CXR_PATH_COL] = self.df[MIMIC_CXR_PATH_COL].apply(
            lambda x: os.path.join(MIMIC_CXR_ROOT_DIR, "/".join(x.split("/")[1:])))
        
        # Load studies and study to text mappings
        self.filenames, self.path2sent = self.load_text_data(split)

        # Filter for split
        self.df = self.df[self.df[MIMIC_CXR_SPLIT_COL] == split]
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.df.reset_index(drop=True, inplace=True)

    def load_text_data(self, split):
        # get study to captions mapping
        # TODO: check this
        filepath = os.path.join(
            BASE_DIR, "../data/captions.pickle")
        if not os.path.isfile(filepath):
            print(
                f"Caption file {filepath} does not exit. Creating captions...")
            path2sent = self.create_path_2_sent_mapping()
            with open(filepath, "wb") as f:
                pickle.dump(path2sent, f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                path2sent = pickle.load(f)

        # filter studies to use for current split
        filenames = []
        for row in self.df.itertuples():
            cur_split = getattr(row, MIMIC_CXR_SPLIT_COL)
            path = getattr(row, MIMIC_CXR_PATH_COL)
            if cur_split == split and path in path2sent:
                filenames.append(path)

        return filenames, path2sent

    def create_path_2_sent_mapping(self):
        sent_lens, num_sents = [], []
        path2sent = {}

        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            # pick impression and findings
            captions = ""
            captions += row["impression"]
            captions += " "
            captions += row["findings"]

            # use space instead of newline
            captions = captions.replace("\n", " ") #captions, aka report findings and impressions

            # split report content into sentences
            captions = captions.split(".")

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions: # for each sentence in report text
                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower()) # word token
                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0: # not null
                        included_tokens.append(t)

                if len(included_tokens) > 0:
                    study_sent.append(" ".join(included_tokens))

                cnt += len(included_tokens)

            if cnt >= 3: #filter out reports with < 3 tokens in total for impressions and findings sections
                sent_lens.append(cnt)
                num_sents.append(len(study_sent))
                path2sent[row[MIMIC_CXR_PATH_COL]] = study_sent

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)

        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent

    def __len__(self):
        return len(self.filenames)
    
    def get_caption(self, path):
        series_sents = self.path2sent[path] #returns list of report sentences

        if len(series_sents) == 0:
            raise Exception("no sentence for path") #pre-processing step failed

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))

        # piece sentences together for each report
        sent = " ".join(series_sents) 

        return sent

    def get_img(self, img_path, scale, transform = None):
        x = cv2.imread(str(img_path), 0)
        # transform images
        x = resize_img(x, scale)
        img = Image.fromarray(x).convert("RGB")
        if transform is not None:
            img = transform(img)
        return img

    def __get_item__(self, index):
        path_key = self.filenames[index]
        report_content = self.get_caption(path_key)
        img = self.get_img(path_key, self.imsize, self.transform)
        return img, report_content, path_key

    
#For testing
if __name__ == "__main__":
    from .transforms import DataTransforms
    transform = DataTransforms(is_train=True)
    dataset = MultimodalPretrainingDataset(split="train", transform=transform)
    data = dataset[0]
    print(data)