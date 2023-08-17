"""
Downstream task: Zero-shot retrieval (Image-to-text & image-to-report)
Label-conditioned retrieval task - positive only when query label matches candidate label

Metric:
- Recall@K
- Precision@K
- Hit@K
where K = {1, 5, 10}
"""

import torch
import torch.nn.functional as F
import argparse
import os
import GPUtil
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, PreTrainedTokenizerFast

import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from evaluation.utils import *
from evaluation.retrieval.retrieval_head import ZeroShotRetrieval
from datasets.dataloader import get_zeroshot_dataloader
from models.builders import *



def get_tokenizer(args):
    if args.model_name in ['gloria', 'medclip', 'convirt']:
        return AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    if args.model_name in ['biovil']:
        return AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True)
    if args.model_name in ['mrm']:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file="/home/faith/projects/unified-framework/models/mrm/mimic_wordpiece.json",
                                            add_special_tokens=True, pad_token='[PAD]')
        return tokenizer


def build_retrieval_model(args, device):
    if args.model_name == "gloria": # uses BioClinicalBERT tokenizer
        gloria_model = load_gloria(args, device=device)
        model = ZeroShotRetrieval(gloria_model.image_encoder_forward, gloria_model.text_encoder_forward, 
                                  get_local_similarities=gloria_model.get_local_similarities, similarity_type=args.similarity_type)
    elif args.model_name == "medclip": # uses BioClinicalBERT tokenizer  
        medclip_model = load_medclip()
        model = ZeroShotRetrieval(medclip_model.encode_image, medclip_model.encode_text, similarity_type=args.similarity_type)
    elif args.model_name == "convirt": # uses BioClinicalBERT tokenizer 
        convirt_model = load_convirt(args)
        model = ZeroShotRetrieval(lambda x: F.normalize(convirt_model.img_encoder.global_embed(
                                                        convirt_model.img_encoder(x)[0]), dim=1), 
                                  lambda x, y, z: F.normalize(convirt_model.text_encoder.global_embed(
                                                              convirt_model.text_encoder(x, y, z)[0]), dim=1),
                                  similarity_type=args.similarity_type)
    elif args.model_name == "biovil": # BiomedVLP-CXR-BERT tokenizer
        biovil_model = load_biovil_model(args, eval=True)
        model = ZeroShotRetrieval(lambda x: biovil_model.get_im_embeddings(x, only_ims=True)[0], 
                                  lambda x, y, z: biovil_model.encode_text(x, y, only_texts=True), 
                                  similarity_type=args.similarity_type)
    elif args.model_name == "mrm": # Custom mimic wordpiece tokenizer
        vision_transformer, bert_model, bert_mlp = load_mrm(args, device=device)
        model = ZeroShotRetrieval(lambda imgs: bert_mlp(vision_transformer.forward_features(imgs))[:, 1:, :].mean(dim=1), 
                                  lambda x, y, z: bert_model(input_ids=x, attention_mask=y, token_type_ids=z)['pooler_output'],
                                  similarity_type = args.similarity_type)
    return model


def get_tokenizer(args):
    if args.model_name in ['gloria', 'medclip', 'convirt']:
        return AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    if args.model_name in ['biovil']:
        return AutoTokenizer.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized", trust_remote_code=True)
    if args.model_name in ['mrm']:
        tokenizer = PreTrainedTokenizerFast(tokenizer_file="/home/faith/unified-framework/models/mrm/mimic_wordpiece.json",
                                            add_special_tokens=True, pad_token='[PAD]')
        return tokenizer
    

def evaluate(predictions, targets):
    hit_at_k, precision_at_k = (), ()

    for k in [1, 5, 10]:
        num_samples = len(predictions)
        num_hits = 0
        num_correct = 0

        for i in range(num_samples): # for each text/image query
            top_k_predictions = predictions[i][:k] # extract top k candidate images/reports
            hit = False
            for idx in top_k_predictions:  # for each candidate
                if torch.equal(targets[idx], targets[i]): # if class of query matches class of candidate 
                    num_correct += 1
                    
                    if hit == False:
                        num_hits += 1
                        hit = True # class of query is found in classes of top K candidates

        hit_frac = num_hits / num_samples
        hit_at_k += (hit_frac,)
        
        precision = num_correct / (num_samples * k)
        precision_at_k += (precision,)

    return {'Hit@K': hit_at_k, 
            'Precision@K': precision_at_k}


def main(args):
    
    # Set up CUDA and GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    args.n_gpu = torch.cuda.device_count()
    torch.set_default_tensor_type('torch.FloatTensor')
    
    # Set seed
    set_seed(args)

    # Build model and tokenizer
    model = build_retrieval_model(args, device).to(device)
    tokenizer = get_tokenizer(args)

    # Get dataset
    dataloader = get_zeroshot_dataloader(args, tokenizer)    

    hits = []
    precisions = []

    # Batch-wise zero-shot image-text retrieval on image-text pairs
    for i, sample in enumerate(tqdm(dataloader)):
        image = sample['image'].to(device)
        text = sample['text']
        for k, v in text.items():
            text[k] = v.to(device)

        targets = sample['target'].to(device)
        predictions = model(image, text, retrieve_images=args.retrieve_images)

        eval_results = evaluate(predictions, targets)
        hit = torch.tensor(eval_results['Hit@K'])
        prec = torch.tensor(eval_results['Precision@K'])

        hits.append(hit)
        precisions.append(prec)

    hits_at_k = torch.stack(hits).mean(dim=0, dtype=float)
    H1, H5, H10 = hits_at_k.tolist()

    precision_at_k = torch.stack(precisions).mean(dim=0, dtype=float)
    P1, P5, P10 = precision_at_k.tolist()

    print(f'Hit@1: {H1}, Hit@5: {H5}, Hit@10: {H10}, \n',
          f'Precision@1: {P1}, Precision@5: {P5}, Precision@10: {P10}')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Customizable model training settings
    parser.add_argument('--dataset', type=str, default='mimic_5x200')
    parser.add_argument('--model_name', type=str, default='', choices=['mrm', 'biovil', 'convirt', 'medclip', 'gloria'])
    parser.add_argument('--similarity_type', default='global', type=str, choices=['global', 'local', 'both'])
    parser.add_argument('--retrieve_images', action='store_true', help='switch to text-to-image retrieval instead of image-to-text')

    # To be configured based on hardware/directory
    parser.add_argument('--pretrain_path', default='Path/To/checkpoint.pth')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=16)
    
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    #torch.cuda.current_device()
    #torch.cuda._initialized = True

    main(args)

