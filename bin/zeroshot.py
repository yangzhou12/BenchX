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


class TextEncoder(nn.Module):
    def __init__(self, language_config):
        super(TextEncoder, self).__init__()

        self.model = EncoderModel(language_config)

        if language_config.language_projection:
            self.projection_head = eval(language_config.language_projection.layer)

            if language_config.pretrained:
                self.projection_head = self.load_pretrained(self.projection_head, language_config.pretrained, language_config.language_projection.prefix)

        self.last_n_layers = language_config.last_n_layers

    def load_pretrained(self, network, pretrain_path, prefix):
        checkpoint = torch.load(pretrain_path, map_location="cpu")

        for key in ["state_dict", "model"]:  # resolve differences in saved checkpoints
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
        
        if prefix:
            state_dict = {k.replace(prefix, ""): v for k, v in checkpoint.items() if prefix in k}
        else:
            state_dict = checkpoint
            print("Checkpoint prefix not set; Full state dictionary returned")
        
        msg = network.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {msg.missing_keys}\nUnexpected keys: {msg.unexpected_keys}")
        
        return network

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids, attention_mask, output_hidden_states=True, mode="text")

        # Aggregate tokens from last n layers
        last_hidden_states = torch.stack(out['hidden_states'][-self.last_n_layers:]) # n_layer, batch, seqlen, emb_dim
        out = last_hidden_states.permute(1, 0, 2, 3).mean(2).mean(1) # pooling

        # Get projection output
        if hasattr(self, "projection_head"):
            out = self.projection_head(out)

        return out


class PromptClassifier(nn.Module):
    def __init__(self, config):
        super(PromptClassifier, self).__init__()
        img_backbone = copy.deepcopy(config.cnn)
        language_backbone = copy.deepcopy(config.language)

        # Initialize backbones
        self.vision_encoder = eval(img_backbone.pop("proto"))(**img_backbone)
        self.language_encoder = TextEncoder(language_backbone)

    def encode_text(self, input_ids=None, attention_mask=None):
        input_ids = input_ids.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        text_embeds = self.language_encoder(input_ids, attention_mask)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds
    
    def encode_image(self, img=None):
        img_embeds = self.vision_encoder(img)
        img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
        return img_embeds

    def compute_logits(self, img_emb, text_emb): # Taken from GLoRIA
        img_emb = img_emb.detach().cpu().numpy()
        text_emb = text_emb.detach().cpu().numpy()
        global_similarities = metrics.pairwise.cosine_similarity(img_emb, text_emb)
        global_similarities = torch.Tensor(global_similarities)
        return global_similarities

    def get_similarities(self, imgs, texts): 
        with torch.no_grad():
            img_embeds = self.encode_image(imgs)
            text_embeds = self.encode_text(texts["input_ids"], texts["attention_mask"])
            #print(img_emb_g.shape, text_emb_g.shape)
            similarities = self.compute_logits(img_embeds, text_embeds)
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
    model = PromptClassifier(model_config).to(device)

    # Process input images and class prompts
    processed_txt = process_class_prompts(cls_prompts, tokenizer, device)
    processed_imgs = process_img(df['Path'].apply(lambda x: os.path.join(args.img_path, x.replace("CheXpert-v1.0-small/", ""))).tolist(), 
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

    # To be configured based on hardware/directory
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    main(args)