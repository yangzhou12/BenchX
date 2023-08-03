import torch
import torch.nn as nn
import copy
from functools import partial
from timm.models.vision_transformer import VisionTransformer
from timm.models.layers import trunc_normal_

import sys
from pathlib import Path
path_root = Path(__file__).parents[0]
sys.path.append(str(path_root))
from biovil.CLIP_Embedding import MedCLIP
from gloria.models.gloria_model import GLoRIA
from convirt.convirt_module import ConVIRT
from medclip.modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT
#from mrm.model_mrm import MRM



def load_biovil_model(args, eval=False):
    biovil_model = MedCLIP(eval=eval)
    biovil_model.cuda()
    
    if args.pretrain_path:
        checkpoint = torch.load(args.pretrain_path, map_location='cpu')
        biovil_model.load_state_dict(checkpoint, strict=False)

    return biovil_model


def build_gloria_encoder(args):
   # load pretrained image encoder
    gloria_model = load_gloria(args)
    image_encoder = copy.deepcopy(gloria_model.img_encoder)
    del gloria_model
    return image_encoder


def load_gloria(args, device="cuda" if torch.cuda.is_available() else "cpu"):
    if args.pretrain_path:
        ckpt = torch.load(args.pretrain_path, map_location="cuda:0")
        cfg = ckpt["hyper_parameters"]
        ckpt_dict = ckpt["state_dict"]

        fixed_ckpt_dict = {}
        for k, v in ckpt_dict.items():
            new_key = k.split("gloria.")[-1]
            fixed_ckpt_dict[new_key] = v
        ckpt_dict = fixed_ckpt_dict

        gloria_model = build_gloria_model(cfg).to(device)
        gloria_model.load_state_dict(ckpt_dict)

        return gloria_model
    else:
        raise RuntimeError("No pretrained weights found!")
        

def build_gloria_model(cfg):
    gloria_model = GLoRIA(cfg)
    return gloria_model


def build_convirt_encoder(args):
    convirt_model = load_convirt(args)
    image_encoder = copy.deepcopy(convirt_model.img_encoder.model)
    del convirt_model
    return image_encoder


def load_convirt(args):
    convirt_model = ConVIRT(img_encoder='resnet_50', emb_dim=128, freeze_bert=True)
    convirt_model.cuda()
    
    if args.pretrain_path:
        checkpoint = torch.load(args.pretrain_path, map_location='cpu')
        convirt_model.load_state_dict(checkpoint, strict=False)

    return convirt_model


def build_mrm_classifier(args, num_classes):

    def vit_base_patch16(**kwargs):
        model = VisionTransformer(norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), **kwargs)
        return model

    model = vit_base_patch16(num_classes=num_classes, drop_path_rate=0.1, global_pool="avg")

    if args.pretrain_path:
        checkpoint_model = torch.load(args.pretrain_path, map_location='cpu')['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        
        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    return model


def load_medclip():
    medclip_model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    medclip_model.from_pretrained()
    medclip_model.cuda()
    return medclip_model


# def load_mrm(args, device="cuda" if torch.cuda.is_available() else "cpu"):
   
#     def vit_base_patch16(**kwargs):
#         model = VisionTransformer(norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), **kwargs)
#         return model

#     mrm_model = vit_base_patch16(num_classes=0, drop_path_rate=0.1, fc_norm=False).to(device)

#     if args.pretrain_path:
#         checkpoint_model = torch.load(args.pretrain_path, map_location='cpu')['model']
#         state_dict = mrm_model.state_dict()
#         for k in ['head.weight', 'head.bias']:
#             if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
#                 print(f"Removing key {k} from pretrained checkpoint")
#                 del checkpoint_model[k]
        
#         # load pre-trained model
#         msg = mrm_model.load_state_dict(checkpoint_model, strict=False)
#         print(msg)

#         assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}
    
#     return mrm_model

    

