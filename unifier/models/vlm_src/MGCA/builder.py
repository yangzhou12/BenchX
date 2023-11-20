import torch
from .models.mgca.mgca_module import MGCA


# Load VLM
def load_mgca(ckpt, **kwargs):
    if ckpt:
        model = MGCA.load_from_checkpoint(ckpt, strict=False)
        return model
    else:
        raise RuntimeError("No pretrained weights found!")