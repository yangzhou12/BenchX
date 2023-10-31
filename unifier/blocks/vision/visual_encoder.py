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
from .huggingface.encoder_model import HFVisionEncoder


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
        return build_model(name=model_name, pretrained=pretrained, **kwargs)
    
    # HuggingFace pretrained model
    elif "hfpretrained" in backbone.lower():
        return HFVisionEncoder(**kwargs)
    
    #####   Models to be loaded manually below   #####

    # Timm Vision Transformer
    elif "timmvit" in backbone.lower():
        network = VisionTransformer(patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias=True,
                                    norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=0, global_pool="", **kwargs)

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
    if pretrained and os.path.exists(pretrained):
        checkpoint = torch.load(pretrained, map_location="cpu")

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


class VisualEncoder(nn.Module):
    def __init__(
        self,
        backbone,
        permute,
        dropout_out=0.0,
        freeze=False,
        output_layer=None,
        pretrained=None,
        prefix=None,
        slice_encode=None,
        slice_dim=None,
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
        self.slice_encode = slice_encode
        self.slice_dim = slice_dim

        if self.slice_encode and self.output_layer == "features":
            raise Exception(
                "If encoding per slices, output of forward pass should be a vector. "
                "Try avgpool or features as output_layer parameter."
            )
        if self.slice_encode and self.slice_dim is None:
            raise Exception(
                "slice_encode is True but slice_dim is None, please specify slice_dim"
            )

        if visual_projection:
            self.visual_projection = nn.Linear(
                visual_projection.in_features, visual_projection.out_features
            )
        else:
            self.visual_projection = nn.Identity()

        assert permute in ["batch_first", "spatial_first", "no_permute"]

        if freeze:
            for name, param in self.model.named_parameters():
                param.requires_grad = False

    def encode(self, images, images_mask=None, **kwargs):
        if torch.cuda.is_available():
            images = images.cuda()
            images_mask = images_mask.cuda() if images_mask is not None else images_mask

        # Single-image forward pass
        if len(images.shape) == 4:
            features = self(images)
            features_mask = torch.sum(torch.abs(features), dim=-1) != 0
            return self.visual_projection(features), features_mask

        # Multi-image or 3D:
        assert len(images.shape) == 5, "wrong images shape"

        # Multi forward pass
        images = rearrange(images, "d0 d1 d2 d3 d4 -> (d0 d1) d2 d3 d4")
        features = self(images)

        if features.ndim <= 2:
            raise Exception(
                "The input size is too small for this model. The spatial dim has been shrunk to 1."
            )

        # Masking features of empty images
        num_images = images.shape[1]
        features = features.view(
            int(features.shape[0] / num_images),
            num_images,
            features.shape[-2],
            features.shape[-1],
        )
        if images_mask is not None:
            features = features * images_mask.unsqueeze(-1).unsqueeze(-1)

        # Creating features-wise attention mask
        features = rearrange(features, "d0 d1 d2 d3 -> d0 (d1 d2) d3")
        features_mask = torch.sum(torch.abs(features), dim=-1) != 0

        return self.visual_projection(features), features_mask

    def forward(self, images, **kwargs):
        out = self.model(images)
        if isinstance(self.model, ViTModel) or isinstance(self.model, DeiTModel):
            assert isinstance(out, BaseModelOutputWithPooling)
            out = self.dropout_out(out.last_hidden_state)
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
            "visual_projection": str(self.visual_projection),
        }
        # Remove None values
        repr_dict = {k: v for k, v in repr_dict.items() if v is not None}

        # Convert to JSON
        repr_json = json.dumps(repr_dict, indent=2)

        return f"{self.backbone}:\n{repr_json}"
