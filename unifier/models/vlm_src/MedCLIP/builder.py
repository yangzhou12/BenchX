from .modeling_medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPVisionModel


# Load VLM (ViT)
def load_medclip_vit(ckpt, **kwargs):
    if ckpt:
        model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        model.from_pretrained(ckpt=ckpt)
        return model
    else:
        raise RuntimeError("No pretrained weights found!")
    

# Load VLM (ResNet-50)
def load_medclip_resnet50(ckpt, **kwargs):
    if ckpt:
        model = MedCLIPModel(vision_cls=MedCLIPVisionModel)
        model.from_pretrained(ckpt=ckpt)
        return model
    else:
        raise RuntimeError("No pretrained weights found!")