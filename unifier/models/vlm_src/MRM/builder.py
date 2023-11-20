import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import VisionTransformer


def load_mrm(ckpt, **kwargs):
    if ckpt:
        def vit_base_patch16(**kwargs):
            model = VisionTransformer(norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
            return model
        
        mrm_model = vit_base_patch16(drop_path_rate=0.1, global_pool="avg", **kwargs)
        checkpoint_model = torch.load(ckpt, map_location="cpu")["model"]
        mrm_model.load_state_dict(checkpoint_model, strict=False)
        return mrm_model
    
    else:
        raise RuntimeError("No pretrained weights found!")
    