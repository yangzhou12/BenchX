import os
from omegaconf import OmegaConf
from unifier.models.vilmedic.ConVIRT import ConVIRT


CONFIG_FILEPATH = "config/templates/pretrain/convirt-mimic_resnet50.yml"

def load_convirt(ckpt, **kwargs):
    if ckpt:
        config = OmegaConf.load(os.path.join(os.getcwd(), CONFIG_FILEPATH))
        model = ConVIRT(**config.model)
        return model
    else:
        raise RuntimeError("No pretrained weights found!")