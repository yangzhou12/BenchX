import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import random
from PIL import Image
import cv2
import copy
from omegaconf import OmegaConf
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import unifier.datasets.transforms as transforms
from unifier.blocks.vision import *
from unifier.blocks.custom.refers.transformer import REFERSViT
from unifier.blocks.huggingface.encoder import EncoderModel

from sklearn import metrics
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, PreTrainedTokenizerFast


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


# CheXpert class prompts
CHEXPERT_CLASS_PROMPTS = {
    "Atelectasis": {
        "severity": ["", "mild", "minimal"],
        "subtype": [
            "subsegmental atelectasis",
            "linear atelectasis",
            "trace atelectasis",
            "bibasilar atelectasis",
            "retrocardiac atelectasis",
            "bandlike atelectasis",
            "residual atelectasis",
        ],
        "location": [
            "at the mid lung zone",
            "at the upper lung zone",
            "at the right lung zone",
            "at the left lung zone",
            "at the lung bases",
            "at the right lung base",
            "at the left lung base",
            "at the bilateral lung bases",
            "at the left lower lobe",
            "at the right lower lobe",
        ],
    },
    "Cardiomegaly": {
        "severity": [""],
        "subtype": [
            "cardiac silhouette size is upper limits of normal",
            "cardiomegaly which is unchanged",
            "mildly prominent cardiac silhouette",
            "portable view of the chest demonstrates stable cardiomegaly",
            "portable view of the chest demonstrates mild cardiomegaly",
            "persistent severe cardiomegaly",
            "heart size is borderline enlarged",
            "cardiomegaly unchanged",
            "heart size is at the upper limits of normal",
            "redemonstration of cardiomegaly",
            "ap erect chest radiograph demonstrates the heart size is the upper limits of normal",
            "cardiac silhouette size is mildly enlarged",
            "mildly enlarged cardiac silhouette, likely left ventricular enlargement. other chambers are less prominent",
            "heart size remains at mildly enlarged",
            "persistent cardiomegaly with prominent upper lobe vessels",
        ],
        "location": [""],
    },
    "Consolidation": {
        "severity": ["", "increased", "improved", "apperance of"],
        "subtype": [
            "bilateral consolidation",
            "reticular consolidation",
            "retrocardiac consolidation",
            "patchy consolidation",
            "airspace consolidation",
            "partial consolidation",
        ],
        "location": [
            "at the lower lung zone",
            "at the upper lung zone",
            "at the left lower lobe",
            "at the right lower lobe",
            "at the left upper lobe",
            "at the right uppper lobe",
            "at the right lung base",
            "at the left lung base",
        ],
    },
    "Edema": {
        "severity": [
            "",
            "mild",
            "improvement in",
            "presistent",
            "moderate",
            "decreased",
        ],
        "subtype": [
            "pulmonary edema",
            "trace interstitial edema",
            "pulmonary interstitial edema",
        ],
        "location": [""],
    },
    "Pleural Effusion": {
        "severity": ["", "small", "stable", "large", "decreased", "increased"],
        "location": ["left", "right", "tiny"],
        "subtype": [
            "bilateral pleural effusion",
            "subpulmonic pleural effusion",
            "bilateral pleural effusion",
        ],
    },
}


class PromptClassifier(nn.Module):
    def __init__(self, 
                 img_encoder_forward, 
                 text_encoder_forward, 
                 get_global_similarities=None, 
                 get_local_similarities=None, 
                 similarity_type="both"):
        
        super(PromptClassifier, self).__init__()
        self.img_encoder_forward = img_encoder_forward
        self.text_encoder_forward = text_encoder_forward
        self.get_local_similarities = get_local_similarities

        if similarity_type not in ["global", "local", "both"]:
            raise RuntimeError(
                "Similarity type should be one of ['global', 'local', 'both']"
            )
        
        if get_local_similarities == None and similarity_type in ["both", "local"]:
            raise RuntimeError(
                "Local similarity function not specified"
            )
        
        self.similarity_type = similarity_type

        if get_global_similarities: # if custom global similarity function exists, use custom function
            self.get_global_similarities = get_global_similarities
        else: # else, use GLoRIA global similarity function
            self.get_global_similarities = self.calc_global_similarities

    def calc_global_similarities(self, img_emb_g, text_emb_g): # Taken from GLoRIA
        img_emb_g = img_emb_g.detach().cpu().numpy()
        text_emb_g = text_emb_g.detach().cpu().numpy()
        global_similarities = metrics.pairwise.cosine_similarity(img_emb_g, text_emb_g)
        global_similarities = torch.Tensor(global_similarities)
        return global_similarities
    
    def get_similarities(self, imgs, texts): 
        with torch.no_grad(): # get image features and compute similarities (global and local)
            if self.get_local_similarities:
                img_emb_l, img_emb_g = self.img_encoder_forward(imgs)
                text_emb_l, text_emb_g, _ = self.text_encoder_forward(texts["input_ids"], texts["attention_mask"], texts["token_type_ids"])
                local_similarities = self.get_local_similarities(img_emb_l, text_emb_l, texts["cap_lens"])
            else:
                img_emb_g = self.img_encoder_forward(imgs)
                outputs = self.text_encoder_forward(texts["input_ids"], texts["attention_mask"], texts["token_type_ids"])
                text_emb_l, text_emb_g = outputs[0], outputs[1]

            #print(img_emb_g.shape, text_emb_g.shape)
            global_similarities = self.get_global_similarities(img_emb_g, text_emb_g)

        if self.similarity_type == "global":
            return global_similarities.detach().cpu().numpy()
        elif self.similarity_type == "local":
            return local_similarities.detach().cpu().numpy()
        else:
            similarities = (local_similarities + global_similarities) / 2 # similarity aggregation function
            return similarities.detach().cpu().numpy()

    def forward(self, img_values=None, prompt_inputs=None):
        class_similarities = []
        class_names = []
        
        for cls_name, cls_text in prompt_inputs.items():
            similarities = self.get_similarities(img_values, cls_text)
            cls_similarity = similarities.max(axis=1) # Take max as the logits similarity
            class_similarities.append(cls_similarity)
            class_names.append(cls_name)
        
        class_similarities = np.stack(class_similarities, 1)

        # standardize across class
        if class_similarities.shape[0] > 1:
            class_similarities = (class_similarities - class_similarities.mean(axis=0)) / (class_similarities.std(axis=0))

        outputs = {
            'logits': class_similarities,
            'class_names': class_names
        }
        return outputs


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def build_prompt_classifier(model_config, device):
    cnn = copy.deepcopy(model_config.cnn)
    lang = copy.deepcopy(model_config.language)
    visual_encoder = eval(cnn.pop("proto"))(**cnn).to(device)
    language_encoder = EncoderModel(lang).to(device)

    model = PromptClassifier(
        visual_encoder.forward,
        language_encoder.forward,
        similarity_type="global"
    )
    return model.to(device)


def get_tokenizer(model_config):
    lang = copy.deepcopy(model_config.language)
    if lang.custom_tokenizer:
        return PreTrainedTokenizerFast(tokenizer_file=lang.tokenizer_file, **lang.custom_tokenizer)
    else:
        return AutoTokenizer.from_pretrained(lang.pop("proto"), trust_remote_code=True)


def generate_chexpert_class_prompts(n = 5):
    prompts = {}    
    for k, v in CHEXPERT_CLASS_PROMPTS.items():
        cls_prompts = []
        keys = list(v.keys())

        # severity
        for k0 in v[keys[0]]:
            # subtype
            for k1 in v[keys[1]]:
                # location
                for k2 in v[keys[2]]:
                    cls_prompts.append(f"{k0} {k1} {k2}")

        if n is not None and n < len(cls_prompts):
            prompts[k] = random.sample(cls_prompts, n)
        else:
            prompts[k] = cls_prompts
    return prompts


def process_class_prompts(cls_prompts, tokenizer, device, max_length=97):
    cls_prompt_inputs = {}

    for cls_name, cls_text in cls_prompts.items():
        text_inputs = tokenizer(
            cls_text,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            max_length=max_length,
        )
        for k, v in text_inputs.items():
            text_inputs[k] = v.to(device)

        # print(cls_text)
        cap_lens = []
        for txt in cls_text:
            cap_lens.append(len([w for w in txt if not w.startswith("[")]))
        text_inputs["cap_lens"] = cap_lens

        cls_prompt_inputs[cls_name] = text_inputs

    return cls_prompt_inputs


def process_img(paths, tfm, device):
    transform = eval("transforms." + tfm)()

    if type(paths) == str:
        paths = [paths]

    all_imgs = []
    for p in paths:
        x = cv2.imread(str(p), 0)
        img = Image.fromarray(x).convert("RGB")
        img = transform(img) # transform images
        all_imgs.append(torch.tensor(img))

    all_imgs = torch.stack(all_imgs).to(device)

    return all_imgs


def main(args):
    # Set up CUDA and GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Total CUDA devices: ", torch.cuda.device_count())

    df = pd.read_csv(_CSVPATH[args.dataset])

    # Set seed
    set_seed(args)

    # Generate input class text prompts
    cls_prompts = generate_chexpert_class_prompts()

    # Build model and tokenizer
    config = OmegaConf.load(args.model_config)
    includes = config.get("includes", [])

    # Loop over includes
    include_mapping = OmegaConf.create()
    for include in includes:
        if not os.path.exists(include):
            include = os.path.join(os.path.dirname(args.config), include)

        current_include_mapping = OmegaConf.load(include)
        include_mapping = OmegaConf.merge(include_mapping, current_include_mapping)

    # Override includes with current config
    config = OmegaConf.merge(include_mapping, config)

    model_config = config.model

    tokenizer = get_tokenizer(model_config)
    model = build_prompt_classifier(model_config, device)

    # Process input images and class prompts
    processed_txt = process_class_prompts(cls_prompts, tokenizer, device)
    processed_imgs = process_img(df['Path'].apply(lambda x: os.path.join(args.img_path, x)).tolist(), 
                                 args.transforms, device)

    output = model(processed_imgs, processed_txt)
    
    y_pred = np.argmax(output['logits'], axis=1)
    y_true = np.argmax(df[CHEXPERT_COMPETITION_TASKS], axis=1)
    accuracy = accuracy_score(y_true, y_pred)

    print(f"Mean accuracy: {accuracy}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Customizable model training settings
    parser.add_argument("--dataset", default="chexpert_5x200", choices=["chexpert_5x200", "mimic_5x200"])
    parser.add_argument("--transforms", type=str, default="DataTransforms")
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--model_config", type=str)
    parser.add_argument("--similarity_type", default="global", type=str, choices=["global", "local", "both"])

    # To be configured based on hardware/directory
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    main(args)