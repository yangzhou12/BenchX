import os
import torch
import json
import torch.nn as nn
from functools import partial
from einops import rearrange
from collections import OrderedDict

from torchvision.models import *

from transformers.models.vit.modeling_vit import ViTModel, ViTConfig
from transformers.models.deit.modeling_deit import DeiTModel, DeiTConfig

from transformers.models.resnet.modeling_resnet import ResNetModel as HFResNetModel

from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    BaseModelOutputWithPoolingAndNoAttention,
    BaseModelOutputWithNoAttention,
)

from transformers import ResNetConfig, PoolFormerConfig
from transformers.models.poolformer.modeling_poolformer import (
    PoolFormerModel as HFPoolFormerModel,
)

from timm.models.vision_transformer import VisionTransformer
from .openai.clip_model import build_model
from .huggingface.encoder_model import HFAutoModel


def load_pretrained(network, pretrain_path, prefix):
    checkpoint = torch.load(pretrain_path, map_location="cpu")

    for key in ["state_dict", "model"]:  # resolve differences in saved checkpoints
        if key in checkpoint:
            checkpoint = checkpoint[key]
            break
    
    if prefix:
        state_dict = {k.replace(prefix, ""): v for k, v in checkpoint.items() if prefix in k}
    else:
        state_dict = checkpoint
        print("Checkpoint prefix not set; Full state dictionary returned")

    for layer_k in ["head.weight", "head.bias", "fc.weight", "fc.bias"]:
        if layer_k in state_dict:
            del state_dict[layer_k]
    
    msg = network.load_state_dict(state_dict, strict=False)
    print(f"Missing keys: {msg.missing_keys}\nUnexpected keys: {msg.unexpected_keys}")
    
    return network


def get_network(backbone, output_layer, pretrained, prefix=None, **kwargs):
    """
    Create sub-network given a backbone and an output_layer with loaded vision weights
    """
    # Create avgpool for densenet, does not exist as such
    if "densenet" in backbone and "3d" not in backbone and output_layer == "avgpool":
        sub_network = get_network(backbone, "features", pretrained, **kwargs)
        sub_network.add_module("relu", nn.ReLU(inplace=True))
        sub_network.add_module("avgpool", nn.AdaptiveAvgPool2d((1, 1)))
        sub_network.add_module("flatten", nn.Flatten(1))
        return sub_network

    # HuggingFace Vision Transformer
    elif "hfvit" in backbone.lower():
        model = ViTModel(ViTConfig(return_dict=True, **kwargs), add_pooling_layer=False)
        return model

    elif "deit" in backbone.lower():
        return DeiTModel(
            DeiTConfig(return_dict=True, **kwargs), add_pooling_layer=False
        )

    # HuggingFace ResNet
    elif "hfresnet" in backbone.lower():
        return HFResNetModel(ResNetConfig(return_dict=True, **kwargs))

    # HuggingFace PoolFormer
    elif "hfpoolformer" in backbone.lower():
        return HFPoolFormerModel(PoolFormerConfig(return_dict=True, **kwargs))
    
    # OpenAI Vision Encoders
    elif "openai" in backbone.lower():
        model_name = backbone.replace("openai_", "")
        return build_model(prefix=prefix, name=model_name, pretrained=pretrained, **kwargs)
    
    # HuggingFace pretrained model
    elif "hfpretrained" in backbone.lower():
        return HFAutoModel(**kwargs)
    
    #####   Models to be loaded manually below   #####

    # Timm Vision Transformer
    elif "timmvit" in backbone.lower():
        network = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=True,
                                    norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    # PyTorch models
    else: 
        is_named = kwargs.pop("named_children", True) # Named layers by default
        weights = get_model_weights(backbone) if (pretrained == "DEFAULT") else None
        network = eval(backbone)(weights=weights, **kwargs)

        if output_layer is not None and (not output_layer == "classifier"):
            layers = [n for n, _ in network.named_children()]
            assert output_layer in layers, "{} not in {}".format(output_layer, layers)

            sub_network = []
            for n, c in network.named_children():
                sub_network.append((n, c))
                if n == output_layer:
                    break

            if is_named:
                network = nn.Sequential(OrderedDict(sub_network))
            else:
                network = nn.Sequential(*list(map(lambda x: x[1], sub_network)))

    # Load custom pretrained weights
    if pretrained not in ["DEFAULT", None]:
        network = load_pretrained(network, pretrained, prefix)

    return network


class VisualEncoder(nn.Module):
    def __init__(
        self,
        backbone,
        permute=None,
        dropout_out=0.0,
        freeze=False,
        output_layer=None,
        pretrained=None,
        prefix=None,
        # slice_encode=None,
        # slice_dim=None,
        visual_projection=None,
        **kwargs,
    ):
        super(VisualEncoder, self).__init__()

        self.backbone = backbone
        self.output_layer = output_layer
        self.permute = permute
        self.freeze = freeze
        self.pretrained = pretrained

        self.model = get_network(self.backbone, self.output_layer, self.pretrained, prefix=prefix, **kwargs)

        self.dropout_out = nn.Dropout(p=dropout_out)

        if visual_projection: # layer, prefix
            self.visual_projection = eval(visual_projection.layer)

            if pretrained not in ["DEFAULT", None]:
                self.visual_projection = load_pretrained(self.visual_projection, pretrained, visual_projection.prefix)
                
        if freeze:
            for name, param in self.model.named_parameters():
                param.requires_grad = False

    def forward(self, images, **kwargs):
        out = self.model(images)

        if isinstance(self.model, ViTModel) or isinstance(self.model, DeiTModel):
            assert isinstance(out, BaseModelOutputWithPooling)
            out = self.dropout_out(out.last_hidden_state)
            # No permute necessary
            return out

        if isinstance(self.model, VisionTransformer):
            out = self.dropout_out(out)
            # No permute necessary
            return out

        if isinstance(self.model, HFResNetModel):
            assert isinstance(out, BaseModelOutputWithPoolingAndNoAttention)
            out = out.last_hidden_state

        if isinstance(self.model, HFPoolFormerModel):
            assert isinstance(out, BaseModelOutputWithNoAttention)
            out = out.last_hidden_state

        out = self.dropout_out(out)

        if self.permute == "no_permute":
            pass
        elif self.permute == "batch_first":
            out = out.view(*out.size()[:2], -1).permute(0, 2, 1)
            if out.shape[1] == 1:  # avgpool case
                out = out.squeeze(1)
        elif self.permute == "spatial_first":
            out = out.view(*out.size()[:2], -1).permute(2, 0, 1)
        else:
            raise NotImplementedError()

        # Visual projection
        if hasattr(self, "visual_projection"):
            out = self.visual_projection(out)

        return out

    def train(self, mode: bool = True):
        if self.freeze:
            mode = False
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def __repr__(self):
        repr_dict = {
            "type": str(type(self.model).__name__)
            if self.backbone.lower() in ["deit", "vit"]
            else None,
            "config": str(self.model.config)
            if self.backbone.lower() in ["deit", "vit"]
            else None,
            "dropout_out": self.dropout_out.p,
            "freeze": self.freeze,
            "output_layer": str(self.output_layer)
            if self.output_layer is not None
            else None,
            "pretrained": self.pretrained
            if self.backbone.lower() not in ["deit", "vit"]
            else None,
            "classifier": str(list(self.model.children())[-1])
            if self.output_layer == "classifier"
            else None,
            "visual_projection": str(self.visual_projection)
            if hasattr(self, "visual_projection")
            else None,
        }
        # Remove None values
        repr_dict = {k: v for k, v in repr_dict.items() if v is not None}

        # Convert to JSON
        repr_json = json.dumps(repr_dict, indent=2)

        return f"{self.backbone}:\n{repr_json}"
