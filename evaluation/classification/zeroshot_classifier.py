import torch
import torch.nn.functional as F
import argparse
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from sklearn.metrics import accuracy_score

import sys

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from evaluation.utils import *
from evaluation.classification.classifier_head import PromptClassifier
from evaluation.classification.prompts import *
from datasets.transforms import DataTransforms, MedCLIPTransforms, GloRIATransforms
from datasets.dataloader import get_zeroshot_dataloader
from models.builders import *


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


def build_prompt_classifier(args, device):
    if args.model_name == "gloria":  # uses BioClinicalBERT tokenizer
        gloria_model = load_gloria(args, device=device)
        model = PromptClassifier(
            gloria_model.image_encoder_forward,
            gloria_model.text_encoder_forward,
            get_local_similarities=gloria_model.get_local_similarities,
            similarity_type=args.similarity_type,
        )
    elif args.model_name == "medclip":  # uses BioClinicalBERT tokenizer
        medclip_model = load_medclip()
        model = PromptClassifier(
            medclip_model.encode_image,
            medclip_model.encode_text,
            get_global_similarities=medclip_model.compute_logits,
            similarity_type=args.similarity_type,
        )
    elif args.model_name == "convirt":  # uses BioClinicalBERT tokenizer
        convirt_model = load_convirt(args)
        model = PromptClassifier(
            lambda x: F.normalize(
                convirt_model.img_encoder.global_embed(convirt_model.img_encoder(x)[0]),
                dim=1,
            ),
            lambda x, y, z: F.normalize(
                convirt_model.text_encoder.global_embed(
                    convirt_model.text_encoder(x, y, z)[0]
                ),
                dim=1,
            ),
            similarity_type=args.similarity_type,
        )
    elif args.model_name == "biovil":  # BiomedVLP-CXR-BERT tokenizer
        biovil_model = load_biovil_model(args, eval=True)
        model = PromptClassifier(
            lambda x: biovil_model.get_im_embeddings(x, only_ims=True)[0],
            lambda x, y, z: biovil_model.encode_text(x, y, only_texts=True),
            similarity_type=args.similarity_type,
        )
    elif args.model_name == "mrm":  # Custom mimic wordpiece tokenizer
        vision_transformer, bert_model, bert_mlp = load_mrm(args, device=device)
        model = PromptClassifier(
            lambda imgs: bert_mlp(vision_transformer.forward_features(imgs))[
                :, 1:, :
            ].mean(dim=1),
            lambda x, y, z: bert_model(input_ids=x, attention_mask=y, token_type_ids=z)[
                "pooler_output"
            ],
            similarity_type=args.similarity_type,
        )
    return model


def get_tokenizer(args):
    if args.model_name in ["gloria", "medclip", "convirt"]:
        return AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    if args.model_name in ['biovil']:
        return AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True)
    if args.model_name in ['mrm']:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file="/home/faith/unified-framework/models/mrm/mimic_wordpiece.json",
                                            add_special_tokens=True, pad_token='[PAD]')
        return tokenizer


def main(args):
    # Set up CUDA and GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Total CUDA devices: ", torch.cuda.device_count())
    args.n_gpu = torch.cuda.device_count()
    torch.set_default_tensor_type("torch.FloatTensor")

    # Set seed
    set_seed(args)

    # Generate input class text prompts
    cls_prompts = generate_chexpert_class_prompts()

    # Build model and tokenizer
    model = build_prompt_classifier(args, device)
    tokenizer = get_tokenizer(args)

    # Get dataset
    dataloader = get_zeroshot_dataloader(args)

    accuracies = []

    for i, sample in enumerate(tqdm(dataloader)):
        image = sample['image'].to(device)
        target = sample['target']
        
        processed_txt = process_class_prompts(cls_prompts, tokenizer, device)

        output = model(image, processed_txt)

        # class_similarities = pd.DataFrame(output['logits'], columns=output['class_names'])
        # print(class_similarities, "\n")

        y_pred = np.argmax(output['logits'], axis=1)
        y_true = np.argmax(target, axis=1)
        accuracy = accuracy_score(y_true, y_pred)

        accuracies.append(accuracy)

    mean_acc = np.stack(accuracies).mean(axis=0, dtype=float)

    print(f"Mean accuracy: {mean_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Customizable model training settings
    parser.add_argument("--dataset", default="chexpert_5x200")
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        choices=["mrm", "biovil", "convirt", "medclip", "gloria"],
    )
    parser.add_argument(
        "--similarity_type",
        default="global",
        type=str,
        choices=["global", "local", "both"],
    )

    # To be configured based on hardware/directory
    parser.add_argument("--pretrain_path", default="Path/To/checkpoint.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gpu", type=str, default="0", help="gpu")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.current_device()
    torch.cuda._initialized = True

    args.phase = "classification"

    main(args)
