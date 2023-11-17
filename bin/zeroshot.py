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
from unifier.blocks.losses.GLoRIALoss import cosine_similarity, gloria_attention_fn

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

        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

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
    
    def aggregate_tokens(self, embeddings, caption_ids):

        batch_size, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []

        # loop over batch
        for embs, caption_id in zip(embeddings, caption_ids):

            agg_embs = []
            token_bank = []
            words = []
            word_bank = []

            # loop over sentence
            for word_emb, word_id in zip(embs, caption_id):

                word = self.idxtoword[word_id.item()]

                if word == "[SEP]":
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))

                    agg_embs.append(word_emb)
                    words.append(word)
                    break

                if not word.startswith("##"):
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                    else:
                        new_emb = torch.stack(token_bank)
                        new_emb = new_emb.sum(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))

                        token_bank = [word_emb]
                        word_bank = [word]
                else:
                    if word.startswith("##"):
                        token_bank.append(word_emb)
                        word_bank.append(word[2:])

            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            paddings = paddings.to(agg_embs.device)
            words = words + ["[PAD]"] * padding_size

            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        return agg_embs_batch, sentences

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = self.model(input_ids, attention_mask, token_type_ids, output_hidden_states=True)

        # Aggregate tokens from last 4 layers (GLoRIA)
        last_hidden_states = torch.stack(out['hidden_states'][-self.last_n_layers:]) # n_layer, batch, seqlen, emb_dim
        out = last_hidden_states.permute(1, 0, 2, 3) #.mean(2).mean(1) # pooling

        embeddings, sents = self.aggregate_tokens(out, input_ids) # agg tokens
        sent_embeds = embeddings.mean(2).mean(1) # global
        word_embeds = embeddings.mean(1) # local
        word_embeds = word_embeds.permute(0, 2, 1)

        # Get projection output
        if hasattr(self, "projection_head"):
            out = self.projection_head(out)

        return word_embeds, sent_embeds


def custom_vision_forward(self, x, **kwargs):
    x = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=True)(x)

    x = self.model.conv1(x)  # (batch_size, 64, 150, 150)
    x = self.model.bn1(x)
    x = self.model.relu(x)
    x = self.model.maxpool(x)

    x = self.model.layer1(x)  # (batch_size, 64, 75, 75)
    x = self.model.layer2(x)  # (batch_size, 128, 38, 38)
    x = self.model.layer3(x)  # (batch_size, 256, 19, 19)
    local_features = x
    x = self.model.layer4(x)  # (batch_size, 512, 10, 10)

    x = self.model.avgpool(x)
    x = x.view(x.size(0), -1)

    # Visual projection
    if hasattr(self, "visual_projection"):
        x = self.visual_projection(x)
    if hasattr(self, "local_projection"):
        local_features = self.local_projection(local_features)

    return local_features, x 


class PromptClassifier(nn.Module):
    def __init__(self, config):
        super(PromptClassifier, self).__init__()
        img_backbone = copy.deepcopy(config.cnn)
        language_backbone = copy.deepcopy(config.language)

        # Initialize backbones
        self.vision_encoder = eval(img_backbone.pop("proto"))(**img_backbone)
        self.language_encoder = TextEncoder(language_backbone)

    def encode_text(self, input_ids=None, attention_mask=None, token_type_ids=None):
        input_ids = input_ids.cuda()
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()
        text_embeds_l, text_embeds_g = self.language_encoder(input_ids, attention_mask, token_type_ids)
        text_embeds_l = text_embeds_l / text_embeds_l.norm(dim=-1, keepdim=True)
        text_embeds_g = text_embeds_g / text_embeds_g.norm(dim=-1, keepdim=True)
        return text_embeds_l, text_embeds_g
    
    def encode_image(self, img=None):
        # add new_forward function to the resnet instance as a class method
        bound_method = custom_vision_forward.__get__(self.vision_encoder, self.vision_encoder.__class__)
        setattr(self.vision_encoder, 'forward', bound_method)
        img_embeds_l, img_embeds_g = self.vision_encoder(img)
        # img_embeds_l = img_embeds_l / img_embeds_l.norm(dim=-1, keepdim=True)
        # img_embeds_g = img_embeds_g / img_embeds_g.norm(dim=-1, keepdim=True)
        return img_embeds_l, img_embeds_g

    def compute_global_similarities(self, img_emb, text_emb): # Taken from GLoRIA
        img_emb = img_emb.detach().cpu().numpy()
        text_emb = text_emb.detach().cpu().numpy()
        global_similarities = metrics.pairwise.cosine_similarity(img_emb, text_emb)
        global_similarities = torch.Tensor(global_similarities)
        return global_similarities
    
    def compute_local_similarities(self, img_emb_l, text_emb_l, cap_lens):
        batch_size = img_emb_l.shape[0]
        similarities = []

        for i in range(len(text_emb_l)):
            words_num = cap_lens[i]
            word = (
                text_emb_l[i, :, 1 : words_num + 1].unsqueeze(0).contiguous()
            )  # [1, 768, 25]

            word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
            context = img_emb_l  # [48, 768, 19, 19]

            weiContext, attn = gloria_attention_fn(
                word, context, 4.0
            )  # [48, 768, 25], [48, 25, 19, 19]

            word = word.transpose(1, 2).contiguous()  # [48, 25, 768]
            weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]

            word = word.view(batch_size * words_num, -1)  # [1200, 768]
            weiContext = weiContext.view(batch_size * words_num, -1)  # [1200, 768]
            #
            row_sim = cosine_similarity(word, weiContext)
            row_sim = row_sim.view(batch_size, words_num)  # [48, 25]

            row_sim.mul_(5.0).exp_()
            row_sim, max_row_idx = torch.max(row_sim, dim=1, keepdim=True)  # [48, 1]

            row_sim = torch.log(row_sim)

            similarities.append(row_sim)

        local_similarities = torch.cat(similarities, 1).detach().cpu()

        return local_similarities

    def get_similarities(self, imgs, texts): 
        with torch.no_grad():
            img_embeds_l, img_embeds_g = self.encode_image(imgs)
            text_embeds_l, text_embeds_g = self.encode_text(texts["input_ids"], texts["attention_mask"], texts["token_type_ids"])

            #print(img_emb_g.shape, text_emb_g.shape)
            global_similarities = self.compute_global_similarities(img_embeds_g, text_embeds_g)
            local_similarities = self.compute_local_similarities(img_embeds_l, text_embeds_l, texts['cap_lens'])
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

    # def _resize_img(img, scale):
    #     """
    #     Args:
    #         img - image as numpy array (cv2)
    #         scale - desired output image-size as scale x scale
    #     Return:
    #         image resized to scale x scale with shortest dimension 0-padded
    #     """
    #     size = img.shape
    #     max_dim = max(size)
    #     max_ind = size.index(max_dim)

    #     # Resizing
    #     if max_ind == 0:
    #         # image is heigher
    #         wpercent = scale / float(size[0])
    #         hsize = int((float(size[1]) * float(wpercent)))
    #         desireable_size = (scale, hsize)
    #     else:
    #         # image is wider
    #         hpercent = scale / float(size[1])
    #         wsize = int((float(size[0]) * float(hpercent)))
    #         desireable_size = (wsize, scale)
    #     resized_img = cv2.resize(
    #         img, desireable_size[::-1], interpolation=cv2.INTER_AREA
    #     )  # this flips the desireable_size vector

    #     # Padding
    #     if max_ind == 0:
    #         # height fixed at scale, pad the width
    #         pad_size = scale - resized_img.shape[1]
    #         left = int(np.floor(pad_size / 2))
    #         right = int(np.ceil(pad_size / 2))
    #         top = int(0)
    #         bottom = int(0)
    #     else:
    #         # width fixed at scale, pad the height
    #         pad_size = scale - resized_img.shape[0]
    #         top = int(np.floor(pad_size / 2))
    #         bottom = int(np.ceil(pad_size / 2))
    #         left = int(0)
    #         right = int(0)
    #     resized_img = np.pad(
    #         resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
    #     )

    #     return resized_img

    all_imgs = []
    for p in paths:
        x = cv2.imread(str(p), 0)       
        # x = _resize_img(x, 256)
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
    parser.add_argument("--similarity_type", default="global", type=str, choices=["global", "local", "both"])

    # To be configured based on hardware/directory
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    main(args)