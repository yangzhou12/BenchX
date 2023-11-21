import torch
from .models.modeling import CONFIGS, VisionTransformer
from .refers.config import Config
from .refers.factories import PretrainingModelFactory


def load_refers(ckpt, **kwargs):
    if ckpt:    
        _C = Config()
        model = PretrainingModelFactory.from_config(_C)
        pretrained_weights = torch.load(ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(pretrained_weights["model"], strict=False)
        return model
    else:
        raise RuntimeError("No pretrained weights found!")