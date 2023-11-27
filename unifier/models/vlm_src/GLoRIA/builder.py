import torch
import torch.nn as nn
import torchvision.transforms as transforms

from . import models


# Load VLM
def load_gloria(ckpt, **kwargs):
    if ckpt:
        gloria_model = build_gloria_from_ckpt(ckpt)
        return gloria_model
    else:
        raise RuntimeError("No pretrained weights found!")


def build_gloria_model(cfg):
    gloria_model = models.gloria_model.GLoRIA(cfg)
    return gloria_model


def build_gloria_from_ckpt(ckpt):
    ckpt = torch.load(ckpt, map_location='cpu')
    cfg = ckpt["hyper_parameters"]
    ckpt_dict = ckpt["state_dict"]

    fixed_ckpt_dict = {}
    for k, v in ckpt_dict.items():
        new_key = k.split("gloria.")[-1]
        fixed_ckpt_dict[new_key] = v
    ckpt_dict = fixed_ckpt_dict

    gloria_model = build_gloria_model(cfg)
    gloria_model.load_state_dict(ckpt_dict, strict=False)

    return gloria_model


def build_img_model(cfg):
    image_model = models.IMAGE_MODELS[cfg.phase.lower()]
    return image_model(cfg)


def build_text_model(cfg):
    return models.text_model.BertEncoder(cfg)


def build_transformation(cfg, split):
    t = []

    if split == "train":

        if cfg.transforms.random_crop is not None:
            t.append(transforms.RandomCrop(cfg.transforms.random_crop.crop_size))

        if cfg.transforms.random_horizontal_flip is not None:
            t.append(
                transforms.RandomHorizontalFlip(p=cfg.transforms.random_horizontal_flip)
            )

        if cfg.transforms.random_affine is not None:
            t.append(
                transforms.RandomAffine(
                    cfg.transforms.random_affine.degrees,
                    translate=[*cfg.transforms.random_affine.translate],
                    scale=[*cfg.transforms.random_affine.scale],
                )
            )

        if cfg.transforms.color_jitter is not None:
            t.append(
                transforms.ColorJitter(
                    brightness=[*cfg.transforms.color_jitter.bightness],
                    contrast=[*cfg.transforms.color_jitter.contrast],
                )
            )
    else:
        if cfg.transforms.random_crop is not None:
            t.append(transforms.CenterCrop(cfg.transforms.random_crop.crop_size))

    t.append(transforms.ToTensor())
    if cfg.transforms.norm is not None:
        if cfg.transforms.norm == "imagenet":
            t.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
        elif cfg.transforms.norm == "half":
            t.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
        else:
            raise NotImplementedError("Normaliation method not implemented")

    return transforms.Compose(t)
