import os
import torch
from omegaconf import OmegaConf
from unifier.models.vilmedic.ConVIRT import ConVIRT


CONFIG_FILEPATH = "config/templates/pretrain/convirt-mimic_resnet50.yml"

def load_convirt(ckpt, **kwargs):
    if ckpt:
        # Build model
        config = OmegaConf.load(os.path.join(os.getcwd(), CONFIG_FILEPATH))
        model = ConVIRT(**config.model)
        
        # Load checkpoint
        ckpt = torch.load(ckpt, map_location='cpu')
        model.load_state_dict(ckpt["model"], strict=False)
        return model
    
    else:
        raise RuntimeError("No pretrained weights found!")