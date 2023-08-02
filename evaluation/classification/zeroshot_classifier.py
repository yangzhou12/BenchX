import torch
import argparse
import os
import random
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report

import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from evaluation.utils import *
from evaluation.classification.classifier_head import PromptClassifier
from models.medclip.prompts import *
from models.medclip.dataset import MedCLIPProcessor
from models.medclip.modeling_medclip import MedCLIPModel, MedCLIPVisionModel


def generate_chexpert_class_prompts(n: int = 5):
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


def process_inputs(args, cls_prompts, img_path_list, device):
    
    if args.model_name == 'gloria':
        gloria_model = load_gloria(args, device=device)
        processed_txt = gloria_model.process_class_prompts(cls_prompts, device)
        processed_imgs = gloria_model.process_img(img_path_list, device)
        zeroshot_model = PromptClassifier(gloria_model.image_encoder_forward, gloria_model.text_encoder_forward, 
                        gloria_model.get_global_similarities, gloria_model.get_local_similarities)

    elif args.model_name == 'medclip':
        processor = MedCLIPProcessor()
        medclip_model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
        medclip_model.from_pretrained()
        processed_txt = process_class_prompts(cls_prompts)
        imgs = img_path_list.map(lambda x: Image.open(x))
        processed_imgs = processor(images=imgs, return_tensors="pt")
        #zeroshot_model = PromptClassifier(medclip_model.encode_image, medclip_model.encode_text, )

    return processed_txt, processed_imgs, zeroshot_model


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
    
    # Process input images and class prompts
    processed_txt, processed_imgs, model = process_inputs(args, cls_prompts, df['Path'].tolist(), device)

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