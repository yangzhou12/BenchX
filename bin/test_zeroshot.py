import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from unifier.models.vlm_src import *
from unifier.blocks.zeroshot import *
from utils import get_seed

from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from utils import get_args

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


def main(args):
    # Set up CUDA and GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Total CUDA devices: ", torch.cuda.device_count())

    # Set seed
    get_seed(seed = args.seed)

    # Generate input class text prompts
    cls_prompts = generate_chexpert_class_prompts(n = 5)
    tokenizer = get_tokenizer(args.tokenizer)
    
    model = build_zeroshot_model(args.model_name, args.ckpt_path, args.mode, device, args.similarity_type)
    model.to(device)

    dataset = eval(args.dataset)(img_path=args.img_path, transform=args.transforms, tokenizer=tokenizer)

    sampler = torch.utils.data.RandomSampler(
        dataset, replacement=False, num_samples=len(dataset)
    )

    zeroshot_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=sampler, 
        shuffle=False,
        collate_fn=dataset.collate_fn if hasattr(dataset, "collate_fn") else None,
        drop_last=False,
    )

    results = []
    for i, sample in enumerate(tqdm(zeroshot_dataloader)):
        image = sample['image'].to(device)
        target = sample['target'].to(device)

        if args.mode == "retrieval":
            text = sample['text'] # throws error if there is no reports for dataset
            text = {k: v.to(device) for k, v in text.items() if type(v) == torch.Tensor}
        elif args.mode == "classification":
            text = process_class_prompts(cls_prompts, tokenizer, device)
        
        output = model(image, text)
        score = model.evaluate(output, target)
        results.append(score)

    scores = {k: np.stack([dic[k] for dic in results]).mean() for k in results[0]}
    for metric, score in scores.items():
        print(f"The average {metric} score is {score}.")


if __name__ == "__main__":
    config, override = get_args()

    main(config)