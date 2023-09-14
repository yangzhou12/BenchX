import os
import torch
import argparse
import shutil
import time
import datetime
import json
import segmentation_models_pytorch as smp
from pathlib import Path
from tqdm import tqdm
from tensorboardX import SummaryWriter

import sys

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from evaluation.utils import *
from datasets.dataloader import get_ft_dataloaders
from evaluation.segmentation.segmentation_loss import *
from evaluation.segmentation.metrics import *
from evaluation.segmentation.transformer_seg import SETRModel
from models.builders import *
from constants import *


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def build_img_segmenter(args):
    """
    Return a image segmenter based on model argument.
    """

    if args.base_model == "resnet50":
        model = smp.Unet("resnet50", encoder_weights=None, activation=None)
        load_encoder_from_checkpoint(args, model.encoder)

        # Freeze encoder
        for param in model.encoder.parameters():
            param.requires_grad = False

    elif args.base_model == "vit":

        def vit_base_patch16(**kwargs):
            model = VisionTransformer(
                norm_layer=partial(torch.nn.LayerNorm, eps=1e-6), **kwargs
            )
            return model

        model = SETRModel(
            patch_size=(16, 16),
            in_channels=3,
            out_channels=1,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            decode_features=[512, 256, 128, 64],
        )
        # create ViT with no classifier and pooling
        encoder = vit_base_patch16(num_classes=0, global_pool="")
        model.encoder_2d.bert_model = encoder

        load_encoder_from_checkpoint(args, model.encoder_2d.bert_model)

        # Freeze encoder
        for param in model.encoder_2d.bert_model.parameters():
            param.requires_grad = False

    else:
        raise RuntimeError(f"Base model name {args.base_model} not found")

    return model


def compute_metrics(
    probability,
    truth,
    num_classes,
    metrics=["medDice", "mIoU", "mDice", "mFscore"],
    threshold=0.5,
):
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert probability.shape == truth.shape

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        p = p.detach().cpu().numpy()
        t = t.detach().cpu().numpy()

        ret_metrics = eval_metrics(p, t, num_classes, metrics=metrics)

    return ret_metrics


def valid(model, data_loader, criterion, device, writer, global_step):
    model.eval()
    val_losses = []
    for i, sample in enumerate(data_loader):
        image = sample["image"]
        label = sample["mask"].float().to(device)
        input_image = image.to(device, non_blocking=True)
        with torch.no_grad():
            pred_class = model(input_image).squeeze(dim=1)
            val_loss = criterion(pred_class, label)
            val_losses.append(val_loss.item())
    avg_val_loss = np.array(val_losses).mean()
    writer.add_scalar("valid/loss", avg_val_loss, global_step)
    return avg_val_loss


def test(args, model, num_classes, data_loader, device, global_step):
    # Initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    model.eval()
    for i, sample in enumerate(data_loader):
        image = sample["image"]
        label = sample["mask"].float().to(device)
        gt = torch.cat((gt, label), 0)
        input_image = image.to(device, non_blocking=True)
        with torch.no_grad():
            logit = model(input_image).squeeze(dim=1)
            prob = torch.sigmoid(logit)
            pred = torch.cat((pred, prob), 0)

    # Compute and print metrics
    ret_metrics = compute_metrics(pred, gt, num_classes)

    tqdm.write("Test results at step {curr_step}:".f