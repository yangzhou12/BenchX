import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import random
from PIL import Image
import cv2
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import unifier.datasets.transforms as transforms
from unifier.models.vlm_src import *

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
            "at the right upper lobe",
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
                 forward_embeddings,
                 global_similarities_func=None,
                 local_similarities_func=None,
                 similarity_type="both"):
        super(PromptClassifier, self).__init__()
        
        # Initialize model's image and text forward methods
        self.forward_embeddings = forward_embeddings

        # Similarity type
        self.similarity_type = similarity_type
        if self.similarity_type not in ["global", "local", "both"]:
            raise NotImplementedError("Similarity type should be one of ['global', 'local', 'both']")
        if local_similarities_func == None and self.similarity_type in ["both", "local"]:
            raise RuntimeError("Local similarity function not specified")

        # Override similarities functions if they exist
        if global_similarities_func:
            self.get_global_similarities = global_similarities_func
        if local_similarities_func:
            self.get_local_similarities = local_similarities_func

    def get_global_similarities(self, img_emb, text_emb): # Taken from GLoRIA
        img_emb = img_emb.detach().cpu().numpy()
        text_emb = text_emb.detach().cpu().numpy()
        global_similarities = metrics.pairwise.cosine_similarity(img_emb, text_emb)
        global_similarities = torch.Tensor(global_similarities)
        return global_similarities

    def get_similarities(self, imgs, texts): 
        with torch.no_grad():
            outputs = self.forward_embeddings(imgs, texts) # imgs - [1000, ...]; texts - [5, ...]
            img_emb_g, text_emb_g = outputs["img_emb_g"], outputs["text_emb_g"]
            global_similarities = self.get_global_similarities(img_emb_g, text_emb_g)

            if hasattr(self, "get_local_similarities"):
                img_emb_l, text_emb_l = outputs["img_emb_l"], outputs["text_emb_l"]
                local_similarities = self.get_local_similarities(img_emb_l, text_emb_l, texts["cap_lens"])
            
        if self.similarity_type == "global":
            return global_similarities.detach().cpu().numpy()
        elif self.similarity_type == "local":
            return local_similarities.detach().cpu().numpy()
        else:
            similarities = (local_similarities + global_similarities) / 2
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


def get_tokenizer(tokenizer, pretrained=False):
    if pretrained:
        return PreTrainedTokenizerFast(tokenizer_file=tokenizer, add_special_tokens=True, pad_token="[PAD]")
    else:
        return AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    

def build_prompt_classifier(model_name, ckpt_path, device, similarity_type="both", **kwargs):
    model = eval(f"load_{model_name}")(ckpt_path, **kwargs).to(device)
    
    local_similarities_func = model.get_local_similarities if hasattr(model, "get_local_similarities") else None

    clf = PromptClassifier(
        forward_embeddings=model.forward_embeddings,
        local_similarities_func=local_similarities_func,
        similarity_type=similarity_type
    )
    return clf


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

        cap_lens = []
        for txt in cls_text:
            cap_lens.append(len([w for w in txt if not w.startswith("[")]))
        text_inputs["cap_lens"] = cap_lens

        cls_prompt_inputs[cls_name] = text_inputs

    return cls_prompt_inputs


def process_img(paths, tfm, device):
    transform = eval("transforms." + tfm)(is_train=False, resize=256, crop_size=224)

    if type(paths) == str:
        paths = [paths]

    all_imgs = []
    for p in paths:
        x = cv2.imread(str(p), 0)       
        img = Image.fromarray(x).convert("RGB")
        img = transform(img) # transform images
        all_imgs.append(img)

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
    cls_prompts = generate_chexpert_class_prompts(n=10)
    tokenizer = get_tokenizer(args.tokenizer, args.pretrained_tokenizer)
    
    clf = build_prompt_classifier(args.model_name, args.ckpt_path, device, args.similarity_type).to(device)

    # Process input images and class prompts
    processed_txt = process_class_prompts(cls_prompts, tokenizer, device)
    processed_imgs = process_img(df['Path'].apply(lambda x: os.path.join(args.img_path, x.replace("CheXpert-v1.0-small/", ""))).tolist(), 
                                 args.transforms, device)

    output = clf(processed_imgs, processed_txt)
    
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
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--tokenizer", type=str, help="file path or huggingface model")
    parser.add_argument("--pretrained_tokenizer", action="store_true")
    parser.add_argument("--similarity_type", type=str, choices=["global", "local", "both"])
    parser.add_argument("--ckpt_path", type=str)

    # To be configured based on hardware/directory
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    main(args)