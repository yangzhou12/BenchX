import os
import torch
import json
import yaml
from .models.tokenization_bert import BertTokenizer
from .models.model_MedKLIP import MedKLIP


DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def get_tokenizer(tokenizer,target_text):
    target_tokenizer = tokenizer(list(target_text), padding='max_length', 
                                 truncation=True, max_length=64, return_tensors="pt")
    return target_tokenizer


def load_medklip(ckpt, **kwargs):
    if ckpt:
        config = yaml.load(open(os.path.join(DIR_PATH, "configs/MedKLIP_config.yaml"), 'r'), Loader=yaml.Loader)

        print("Creating book")
        json_book = json.load(open(os.path.join(DIR_PATH, config['disease_book']), 'r'))
        disease_book = [json_book[i] for i in json_book]
        tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
        disease_book_tokenizer = get_tokenizer(tokenizer,disease_book)
        
        print("Creating model")
        model = MedKLIP(config, disease_book_tokenizer)

        print('Load model from checkpoint')
        checkpoint = torch.load(ckpt, map_location='cpu')
        state_dict = checkpoint['model']          
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()} # prefix change
        model.load_state_dict(state_dict)
        return model
    else:
        raise RuntimeError("No pretrained weights found!")
    