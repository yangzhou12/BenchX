import torch
from .utils.utils_builder import ResNet_CXRBert


def load_mflag(ckpt, **kwargs):
    if ckpt:    
        model = ResNet_CXRBert()
        checkpoint = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(checkpoint)
        return model
    else:
        raise RuntimeError("No pretrained weights found!")