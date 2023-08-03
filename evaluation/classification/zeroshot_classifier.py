import torch
import torch.nn.functional as F
import argparse
import os
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer
from sklearn.metrics import classification_report

import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from evaluation.utils import *
from evaluation.classification.classifier_head import PromptClassifier
from evaluation.classification.prompts import *
from datasets.transforms import DataTransforms
from models.builders import *



def process_class_prompts(cls_prompts, tokenizer, device, max_length=97):
    cls_prompt_inputs = {}
    
    for cls_name, cls_text in cls_prompts.items():

        text_inputs = tokenizer(cls_text, truncation=True, padding="max_length", return_tensors='pt', max_length=max_length)
        for k, v in text_inputs.items():
            text_inputs[k] = v.to(device)

        # print(cls_text)
        cap_lens = []
        for txt in cls_text:
            cap_lens.append(len([w for w in txt if not w.startswith("[")]))
        text_inputs['cap_lens'] = cap_lens

        cls_prompt_inputs[cls_name] = text_inputs

    return cls_prompt_inputs


def process_imgs(paths, device):
    transform = DataTransforms(is_train=False)
    
    if type(paths) == str: #input is one img path
        paths = [paths]

    all_imgs = []
    for p in paths:

        x = cv2.imread(str(p), 0)

        # transform images
        img = Image.fromarray(x).convert("RGB")
        img = transform(img)
        all_imgs.append(torch.tensor(img))

    all_imgs = torch.stack(all_imgs).to(device)

    return all_imgs


def build_prompt_classifier(args, device):
    if args.model_name == "gloria":
        gloria_model = load_gloria(args, device=device)
        model = PromptClassifier(gloria_model.image_encoder_forward, gloria_model.text_encoder_forward, 
                                 get_local_similarities=gloria_model.get_local_similarities, similarity_type=args.similarity_type)
    elif args.model_name == "medclip":
        medclip_model = load_medclip()
        model = PromptClassifier(medclip_model.encode_image, medclip_model.encode_text, similarity_type=args.similarity_type)
    elif args.model_name == "convirt":
        convirt_model = load_convirt(args)
        model = PromptClassifier(lambda x: F.normalize(convirt_model.img_encoder.global_embed(
                                                       convirt_model.img_encoder(x)[0]), dim=1), 
                                 lambda x, y, z: F.normalize(convirt_model.text_encoder.global_embed(
                                                             convirt_model.text_encoder(x, y, z)[0]), dim=1),
                                 similarity_type=args.similarity_type
                                )
    elif args.model_name == "biovil":
        biovil_model = load_biovil_model(args, eval=True)
        model = PromptClassifier(lambda x: biovil_model.get_im_embeddings(x, only_ims=True)[0], 
                                 lambda x, y, z: biovil_model.encode_text(x, y, only_texts=True), 
                                 similarity_type=args.similarity_type)
    #elif args.model_name == "mrm":
    #    mrm_model = load_mrm(args, device=device)
    #    model = PromptClassifier()
    return model


def main(args):
    
    # Set up CUDA and GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    args.n_gpu = torch.cuda.device_count()
    torch.set_default_tensor_type('torch.FloatTensor')
    
    # Set seed
    set_seed(args)

    df = pd.read_csv(CHEXPERT_5x200)
    
    # Generate input class text prompts
    cls_prompts = generate_chexpert_class_prompts()
    
    # Build model and tokenizer
    model = build_prompt_classifier(args, device)
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    # Process input images and class prompts
    processed_txt = process_class_prompts(cls_prompts, tokenizer, device)
    processed_imgs = process_imgs(df['Path'].tolist(), device)

    # Zero-shot classification on 1000 images
    output = model(processed_imgs, processed_txt)

    class_similarities = pd.DataFrame(output['logits'], columns=output['class_names'])
    print(class_similarities, "\n")

    df_5x200 = df[output['class_names']]
    y_pred = np.argmax(output['logits'], axis=1)
    y_true = np.argmax(df_5x200, axis=1)
    print(classification_report(y_true, y_pred))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Customizable model training settings
    parser.add_argument('--model_name', type=str, default='', help='model name')
    parser.add_argument('--similarity_type', default='global', type=str, choices=['global', 'local', 'both'])

    # To be configured based on hardware/directory
    parser.add_argument('--pretrain_path', default='Path/To/checkpoint.pth')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.current_device()
    torch.cuda._initialized = True

    main(args)