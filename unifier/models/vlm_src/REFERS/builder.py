import torch
from .models.modeling import CONFIGS, VisionTransformer


def load_refers(ckpt, **kwargs):
    if ckpt:    
        config = CONFIGS["ViT-B_16"]
        model = VisionTransformer(config, 224, zero_head=True, **kwargs) # num_classes

        pretrained_weights = torch.load(ckpt, map_location=torch.device('cpu'))
        model_weights = model.state_dict()
        load_weights = {k: v for k, v in pretrained_weights.items() if k in model_weights}

        print("load weights")
        model_weights.update(load_weights)
        model.load_state_dict(model_weights)
        return model
    
    else:
        raise RuntimeError("No pretrained weights found!")