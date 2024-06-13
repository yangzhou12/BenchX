# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os.path as osp
from collections import OrderedDict

import mmengine
import torch
from mmengine.runner import CheckpointLoader

def convert_vit(ckpt, prefix=""):
    if len(prefix) > 0 and not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)

    ckpt = {
        k[prefix_len:]: v
        for k, v in ckpt.items() if k.startswith(prefix)
    }

    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith('decoder'):
            continue
        if k.startswith('bert'):
            continue
        if k.startswith('mask'):
            continue
        if k.startswith('norm'):
            new_k = k.replace('norm.', 'ln1.')
        elif k.startswith('norm1'):
            new_k = k.replace('norm1.', 'ln1.')
        elif k.startswith('norm2'):
            new_k = k.replace('norm2.', 'ln2.')
        elif k.startswith('patch_embed'):
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        elif k.startswith('blocks'):
            if 'norm1' in k:
                new_k = k.replace('norm1', 'ln1')
            elif 'norm2' in k:
                new_k = k.replace('norm2', 'ln2')
            elif 'mlp.fc1' in k:
                new_k = k.replace('mlp.fc1', 'ffn.layers.0.0')
            elif 'mlp.fc2' in k:
                new_k = k.replace('mlp.fc2', 'ffn.layers.1')
            elif 'attn.qkv' in k:
                new_k = k.replace('attn.qkv.', 'attn.attn.in_proj_')
            elif 'attn.proj' in k:
                new_k = k.replace('attn.proj', 'attn.attn.out_proj')
            else:
                new_k = k
            new_k = new_k.replace('blocks.', 'layers.')
        else:
            new_k = k
        new_ckpt[new_k] = v

    return new_ckpt

def convert_ptunifier(ckpt, prefix=""):
    if len(prefix) > 0 and not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)

    ckpt = {
        k[prefix_len:]: v
        for k, v in ckpt.items() if k.startswith(prefix)
    }

    new_ckpt = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith("class_embedding"):
            new_k = k.replace("class_embedding", "cls_token")
        elif k.startswith( "positional_embedding"):
            new_k = k.replace("positional_embedding", "pos_embed")
        elif k.startswith("ln_post"):
            new_k = k.replace("ln_post", "ln1")
        elif k.startswith("conv1"):
            new_k = k.replace("conv1", "patch_embed.projection")
        elif k.startswith("transformer.resblocks"):
            if "ln_" in k:
                new_k = k.replace("ln_", "ln")
            elif "mlp.c_fc" in k:
                new_k = k.replace("mlp.c_fc", "ffn.layers.0.0")
            elif "mlp.c_proj" in k:
                new_k = k.replace("mlp.c_proj", "ffn.layers.1")
            elif 'attn' in k:
                new_k = k.replace('attn.', 'attn.attn.')
            else:
                new_k = k
            new_k = new_k.replace("transformer.resblocks", "layers")
        else:
            new_k = k
        new_ckpt[new_k] = v

    new_ckpt['cls_token'] = new_ckpt['cls_token'].unsqueeze(0).unsqueeze(0)
    new_ckpt['pos_embed'] = new_ckpt['pos_embed'].unsqueeze(0)
    return new_ckpt

def convert_refers(ckpt, prefix=""):
    if len(prefix) > 0 and not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)

    ckpt = {
        k[prefix_len:]: v
        for k, v in ckpt.items() if k.startswith(prefix)
    }

    new_ckpt = OrderedDict()

    for k, v in ckpt.items():
        if k.startswith('embeddings'):
            if "position_embeddings" in k:
                new_k = k.replace('embeddings.position_embeddings', 'pos_embed')
            elif "patch_embeddings" in k:
                new_k = k.replace("embeddings.patch_embeddings", "patch_embed.projection")
            elif "cls_token" in k:
                new_k = k.replace("embeddings.cls_token", "cls_token")
        elif k.startswith('encoder'):
            if "attention_norm" in k:
                new_k = k.replace("attention_norm.", "ln1.")
            elif "encoder_norm" in k:
                new_k = k.replace("encoder.encoder_norm.", "ln1.")
            elif "ffn_norm" in k:
                new_k = k.replace("ffn_norm.", "ln2.")
            # elif "attn.out" in k:
            #     new_k = k.replace("attn.out.", "attn.proj.")
            elif "attn.out" in k:
                new_k = k.replace("attn.out.", "attn.attn.out_proj.")
            elif "ffn.fc1" in k:
                new_k = k.replace("ffn.fc1", "ffn.layers.0.0")
            elif "ffn.fc2" in k:
                new_k = k.replace("ffn.fc2", "ffn.layers.1")
            else:
                new_k = k
            new_k = new_k.replace('encoder.layer.', 'layers.')
        else:
            new_k = k
        new_ckpt[new_k] = v

    num_layers = 12
    for i in range(num_layers):
        prefix = f"layers.{i}.attn."
        key = prefix + "query.weight"
        q_weight = new_ckpt.pop(key)
        key = prefix + "key.weight" 
        k_weight = new_ckpt.pop(key)
        key = prefix + "value.weight"
        v_weight = new_ckpt.pop(key)

        key = prefix + "query.bias"
        q_bias = new_ckpt.pop(key)
        key = prefix + "key.bias"
        k_bias = new_ckpt.pop(key)
        key = prefix + "value.bias"
        v_bias = new_ckpt.pop(key)

        weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
        bias = torch.cat((q_bias, k_bias, v_bias), dim=0)

        # new_ckpt[prefix + "qkv.weight"] = weight
        # new_ckpt[prefix + "qkv.bias"] = bias

        new_ckpt[prefix + "attn.in_proj_weight"] = weight
        new_ckpt[prefix + "attn.in_proj_bias"] = bias

    return new_ckpt


def convert_resnet(state_dict, prefix):
    if not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)

    state_dict = {
        k[prefix_len:]: v
        for k, v in state_dict.items() if k.startswith(prefix)
    }

    return state_dict

def convert_medklip(state_dict, prefix):
    if not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)

    state_dict = {
        k[prefix_len:]: v
        for k, v in state_dict.items() if k.startswith(prefix)
    }

    new_ckpt = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith("0"):
            new_k = k.replace("0.", "conv1.")
        elif k.startswith( "1"):
            new_k = k.replace("1.", "bn1.")
        elif k.startswith("4"):
            new_k = k.replace("4.", "layer1.")
        elif k.startswith("5"):
            new_k = k.replace("5.", "layer2.")
        elif k.startswith("6"):
            new_k = k.replace("6.", "layer3.")
        else:
            new_k = k
        new_ckpt[new_k] = v
    return new_ckpt

def convert_swin(ckpt, prefix=""):
    if len(prefix) > 0 and not prefix.endswith('.'):
        prefix += '.'
    prefix_len = len(prefix)

    ckpt = {
        k[prefix_len:]: v
        for k, v in ckpt.items() if k.startswith(prefix)
    }

    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1,
                                            2).reshape(out_channel, in_channel)
        return x

    def correct_unfold_norm_order(x):
        in_channel = x.shape[0]
        x = x.reshape(4, in_channel // 4)
        x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
        return x

    new_ckpt = OrderedDict()

    for key in list(ckpt.keys()):
        new_key = key.replace("embeddings.patch_embeddings", "patch_embed")
        new_key = new_key.replace("embeddings.norm", "patch_embed.norm")
        new_key = new_key.replace("encoder.layers", "stages")
        new_key = new_key.replace("layernorm.", "norm3.")
        new_key = new_key.replace("layernorm_before", "norm1")
        new_key = new_key.replace("layernorm_after", "norm2")
        new_key = new_key.replace("attention.self", "attn.w_msa")
        new_key = new_key.replace("attention.output.dense", "attn.w_msa.proj")
        new_key = new_key.replace("intermediate.dense", "ffn.layers.0.0")
        new_key = new_key.replace("output.dense", "ffn.layers.1")
        new_ckpt[new_key] = ckpt.pop(key)

    depths = [2, 2, 6, 2]
    for i in range(len(depths)):
        for j in range(depths[i]):
            prefix = f"stages.{i}.blocks.{j}.attn.w_msa."
            key = prefix + "query.weight"
            q_weight = new_ckpt.pop(key)
            key = prefix + "key.weight" 
            k_weight = new_ckpt.pop(key)
            key = prefix + "value.weight"
            v_weight = new_ckpt.pop(key)

            key = prefix + "query.bias"
            q_bias = new_ckpt.pop(key)
            key = prefix + "key.bias"
            k_bias = new_ckpt.pop(key)
            key = prefix + "value.bias"
            v_bias = new_ckpt.pop(key)

            weight = torch.cat((q_weight, k_weight, v_weight), dim=0)
            bias = torch.cat((q_bias, k_bias, v_bias), dim=0)

            new_ckpt[prefix + "qkv.weight"] = weight
            new_ckpt[prefix + "qkv.bias"] = bias

    for k, v in new_ckpt.items():
        if 'downsample' in k:
            new_k = k
            if 'reduction.' in k:
                new_v = correct_unfold_reduction_order(v)
            elif 'norm.' in k:
                new_v = correct_unfold_norm_order(v)
            new_ckpt[new_k] = new_v
    return new_ckpt

def main():
    parser = argparse.ArgumentParser(
        description='Convert keys in MedVLP models to the MMSegmentation format.')

    parser.add_argument('src_dir', default="<path_to_source_ckpts>", nargs='?', help='src model path or url')
    # The dst path must be a full path of the new checkpoint.
    parser.add_argument('dst_dir', default="<path_to_target_ckpts>", nargs='?', help='save path')
    args = parser.parse_args()

    support_models = {
        "ConVIRT": {"ckpt_name": "ConVIRT.pth", "prefix": "visual.model"},
        "GLoRIA": {"ckpt_name": "GLoRIA.ckpt", "prefix": "gloria.img_encoder.model"},
        "MedCLIP-R50": {"ckpt_name": "MedCLIP-resnet50.bin", "prefix": "vision_model.model"},
        "MedCLIP-ViT": {"ckpt_name": "MedCLIP-vit.bin", "prefix": "vision_model.model"},
        "MedKLIP": {"ckpt_name": "MedKLIP.pth", "prefix": "module.res_features"},
        "M-FLAG": {"ckpt_name": "M-FLAG.pth", "prefix": "encoder"},
        "MGCA-R50": {"ckpt_name": "MGCA-resnet50.ckpt", "prefix": "img_encoder_q.model"},
        "MGCA-ViT": {"ckpt_name": "MGCA-vit.ckpt", "prefix": "img_encoder_q.model"},
        "MRM": {"ckpt_name": "MRM.pth", "prefix": "vision_encoder.visual"},
        "PTUnifier": {"ckpt_name": "PTUnifier.ckpt", "prefix": ""},
        "REFERS": {"ckpt_name": "REFERS.pth", "prefix": "visual.vit"},
    }    

    for model, ckpt_info in support_models.items():
        source_ckpt_path = osp.join(args.src_dir, ckpt_info["ckpt_name"])
        target_ckpt_path = osp.join(args.dst_dir, ckpt_info["ckpt_name"])
        prefix = ckpt_info["prefix"]

        checkpoint = CheckpointLoader.load_checkpoint(source_ckpt_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            # timm checkpoint
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            # deit checkpoint
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        if model in ["ConVIRT", "GLoRIA", "MGCA-R50", "MedCLIP-R50", "M-FLAG"]:
            weight = convert_resnet(state_dict, prefix=prefix)
        elif model in ["MedKLIP"]:
            weight = convert_medklip(state_dict, prefix=prefix)
        elif model in ["PTUnifier"]:
            weight = convert_ptunifier(state_dict, prefix=prefix)
        elif model in ["MedCLIP-ViT"]:
            weight = convert_swin(state_dict, prefix)
        elif model in ["MGCA-ViT", "MRM"]:
            weight = convert_vit(state_dict, prefix)
        elif model in ["REFERS"]:
            weight = convert_refers(state_dict, prefix)
        
        mmengine.mkdir_or_exist(osp.dirname(target_ckpt_path))
        torch.save(weight, target_ckpt_path)


if __name__ == '__main__':
    main()
