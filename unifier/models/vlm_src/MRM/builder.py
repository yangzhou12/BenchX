import torch
from . import model_mrm


def load_mrm(ckpt, **kwargs):
    if ckpt:
        model = model_mrm.__dict__['mrm']()
        checkpoint = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        return model
    else:
        raise RuntimeError("No pretrained weights found!")
    