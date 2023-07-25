from functools import partial
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer


######################################
#         Vision Transformers        #
######################################

'''
    Set num_classes to zero for classification head to be Identity layer.
'''

def vit_base_patch16(**kwargs):
    model = VisionTransformer(num_classes=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_base_patch16_singlechannel(**kwargs):
    model = VisionTransformer(num_classes=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_large_patch16(**kwargs):
    model = VisionTransformer(num_classes=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_huge_patch14(**kwargs):
    model = VisionTransformer(num_classes=0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model