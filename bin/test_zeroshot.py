import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import argparse
import random
from PIL import Image
import cv2
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from unifier.models.vlm_src import *
from unifier.blocks.zeroshot import *
from utils import get_seed

from sklearn import metrics
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def get_tokenizer(tokenizer):
    if os.path.exists(tokenizer): # File path
        return PreTrainedTokenizerFast(tokenizer_file=tokenizer, add_special_tokens=True, pad_token="[PAD]")
    else: # Huggingface 
        return AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    

def build_zeroshot_model(model_name, ckpt_path, mode, device, similarity_type="both", **kwargs):
    model = eval(f"load_{model_name}")(ckpt_path, **kwargs).to(device)
    
    custom_forward_func = model.zeroshot_forward if hasattr(model, "zeroshot_forward") else None
    if custom_forward_func:
        return ZeroshotModel(
            zeroshot_forward=custom_forward_func,
            mode=mode
        )

    local_similarities_func = model.get_local_similarities if hasattr(model, "get_local_similarities") else None
    model = ZeroshotModel(
        zeroshot_forward=custom_forward_func,
        forward_embeddings=model.forward_embeddings,
        local_similarities_func=local_similarities_func,
        similarity_type=similarity_type,
        mode=mode
    )
    return model


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


def process_report(report_text, tokenizer, device, max_length=97):
    text_inputs = tokenizer(report_text, truncation=True, padding="max_length", 
                                    return_tensors='pt', max_length=max_length)
    for k, v in text_inputs.items():
        text_inputs[k] = v.to(device)

    cap_lens = []
    for txt in report_text:
        cap_lens.append(len([w for w in txt if not w.startswith("[")]))
    text_inputs["cap_lens"] = cap_lens

    return text_inputs


def process_img(paths, tfm, device):
    transform = eval("transforms." + tfm)(is_train=False)

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
    get_seed(seed = args.seed)

    # Generate input class text prompts
    cls_prompts = generate_chexpert_class_prompts(n = 5)
    tokenizer = get_tokenizer(args.tokenizer)
    
    model = build_zeroshot_model(args.model_name, args.ckpt_path, args.mode, device, args.similarity_type)
    model.to(device)

    dataset = eval(args.dataset)(img_path=args.img_path, transform=args.transforms, tokenizer=tokenizer)

    zeroshot_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
        drop_last=False,
    )

    results = []
    for i, sample in enumerate(tqdm(zeroshot_dataloader)):
        image = sample['image']
        target = sample['target']

        if args.mode == "retrieval":
            text = sample['text'] # throws error if there is no reports for dataset
        elif args.mode == "classification":
            text = process_class_prompts(cls_prompts, tokenizer, device)
        
        output = model(image, text)
        score = model.evaluate(output, target)
        results.append(score)

    scores = {k: np.stack([dic[k] for dic in results]).mean() for k in results[0]}
    for metric, score in scores.items():
        print(f"The average {metric} score is {score}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Customizable model training settings
    parser.add_argument("--dataset", default="CheXpert_5x200", choices=["CheXpert_5x200", "MIMIC_5x200"])
    parser.add_argument("--transforms", type=str, default="DataTransforms")
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--tokenizer", type=str, help="file path or huggingface model")
    parser.add_argument("--similarity_type", type=str, choices=["global", "local", "both"])
    parser.add_argument("--mode", type=str, default="retrieval", choices=["retrieval", "classification"])
    parser.add_argument("--batch_size", default=1000)

    # To be configured based on hardware/directory
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    main(args)