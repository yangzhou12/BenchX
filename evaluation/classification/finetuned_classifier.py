import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import argparse
import time
import json
import yaml
import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path
from tensorboardX import SummaryWriter
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score

from evaluation.utils import *
from evaluation.classification.classifier_head import ImageClassifier
from datasets.dataloader import get_ft_dataloaders
from models import builders


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def build_img_classifier(args, num_classes):
    """
    Return a image classifier based on model argument.
    """

    if args.model_name not in available_models():
        raise RuntimeError(f"Model {args.model_name} not supported")

    model_name, backbone = args.model_name.split('-')

    if backbone == "resnet50":
        img_backbone = getattr(builders, f'build_{model_name}_encoder')(args)
        img_classifier = ImageClassifier(backbone=img_backbone, num_classes=num_classes)
    elif backbone == "vit":
        img_classifier = getattr(builders, f'build_{model_name}_classifier')(args, num_classes)
    else:
        raise RuntimeError(f"Backbone {backbone} not found")
    
    return img_classifier


def compute_AUCs(gt, pred, n_class):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
        true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
        can either be probability estimates of the positive class,
        confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(n_class):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    return AUROCs


def valid(model, data_loader, criterion, device, writer, global_step):
    model.eval()
    val_losses = []
    for i, sample in enumerate(data_loader):
        image = sample["image"]
        label = sample["label"].float().to(device)
        input_image = image.to(device, non_blocking=True)
        with torch.no_grad():
            pred_class = model(input_image)
            val_loss = criterion(pred_class, label)
            val_losses.append(val_loss.item())
    avg_val_loss = np.array(val_losses).mean()
    writer.add_scalar("valid/loss", avg_val_loss, global_step)
    return avg_val_loss


def test(args, model, data_loader, device, classes):
    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    model.eval()
    for i, sample in enumerate(data_loader):
        image = sample["image"]
        label = sample["label"].float().to(device)
        gt = torch.cat((gt, label), 0)
        input_image = image.to(device, non_blocking=True)
        with torch.no_grad():
            pred_class = model(input_image)
            pred_class = F.sigmoid(pred_class)
            pred = torch.cat((pred, pred_class), 0)

    # Compute AUROCs
    num_classes = len(classes)
    AUROCs = compute_AUCs(gt, pred, num_classes)
    AUROC_avg = np.array(AUROCs).mean()

    # Compute F1 scores and accuracy
    gt_np = gt[:, 0].cpu().numpy()
    pred_np = pred[:, 0].cpu().numpy()
    precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(
        numerator, denom, out=np.zeros_like(denom), where=(denom != 0)
    )
    max_f1 = np.max(f1_scores)
    max_f1_thresh = thresholds[np.argmax(f1_scores)]

    # Print metrics
    tqdm.write("The max f1 is", max_f1)
    tqdm.write("The accuracy is", accuracy_score(gt_np, pred_np > max_f1_thresh))
    tqdm.write("The average AUROC is {AUROC_avg:.3f}".format(AUROC_avg=AUROC_avg))
    for i in range(num_classes):
        tqdm.write("The AUROC of {} is {}".format(classes[i], AUROCs[i]))

    return AUROC_avg


def train(args, model, exp_path, device):

    # Load specified dataset
    print("Creating dataset")
    train_dataloader, val_dataloader, test_dataloader = get_ft_dataloaders(args)
    classes = DATASET_CLASSES[args.dataset]

    # Prepare loss
    criterion = nn.BCEWithLogitsLoss()

    # Training steps
    print("Start training")
    start_time = time.time()
    best_val_loss = 10000
    writer = SummaryWriter(os.path.join(exp_path, "log"))

    # Save params in txt file
    save_params(args, exp_path)

    # Train by batch
    t_total = args.num_steps
    global_step = 0

    global_step, optimizer, lr_scheduler = load_training_setup(args, exp_path, model)

    while True:
        model.train()

        epoch_iterator = tqdm(
            train_dataloader,
            desc="Training (X / X Steps) (loss=X.X)",
            bar_format="{l_bar}{r_bar}",
            dynamic_ncols=True,
        )

        for i, sample in enumerate(epoch_iterator):
            image = sample["image"]
            label = sample["label"].float().to(device)  # batch_size, num_class
            input_image = image.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred_class = model(input_image)  # batch_size, num_class

            loss = criterion(pred_class, label)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad
            )  # Tackle exploding gradients
            optimizer.step()
            lr_scheduler.step(global_step)
            global_step += 1

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, loss)
            )
            writer.add_scalar("train/loss", loss, global_step)
            writer.add_scalar(
                "train/lr", lr_scheduler._get_lr(global_step)[0], global_step
            )

            # Validate every T steps, as specified by val_step
            if args.val_steps >= 0:
                val_step = args.val_steps
            else:
                val_step = len(train_dataloader)

            if global_step % val_step == 0:
                epoch_iterator.set_description(
                    "Validating... (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, loss)
                )
                
                val_loss = valid(
                    model, val_dataloader, criterion, device, writer, global_step
                )

                # Log training and validation stats at validation step
                log_stats = {
                    "loss": loss.item(),
                    "val_loss": val_loss.item(),
                    "lr": lr_scheduler._get_lr(global_step)[0],
                    "step": global_step,
                }
                with open(os.path.join(exp_path, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                save_obj = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "step": global_step,
                }

                # Save most recent tested checkpoint at each specified validation interval
                torch.save(save_obj, os.path.join(exp_path, "checkpoint_state.pth"))

                # Save and test model with best validation score
                if val_loss < best_val_loss:
                    torch.save(save_obj, os.path.join(exp_path, "best_valid.pth"))
                    best_val_loss = val_loss

                    epoch_iterator.set_description(
                        "Testing... (%d / %d Steps) (val_loss=%2.5f)" % (global_step, t_total, val_loss)
                    )
                    test_auc = test(args, model, test_dataloader, device, classes)

                    with open(os.path.join(exp_path, "log.txt"), "a") as f:
                        f.write(
                            "The average AUROC is {AUROC_avg:.4f}".format(
                                AUROC_avg=test_auc
                            )
                            + "\n"
                        )

            if global_step % t_total == 0:
                break

        if global_step % t_total == 0:
            break

    writer.close()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("End training!")
    print("Training time {}".format(total_time_str))


def main(args):
    # Set up CUDA and GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Total CUDA devices: ", torch.cuda.device_count())
    args.n_gpu = torch.cuda.device_count()
    torch.set_default_tensor_type("torch.FloatTensor")

    # Set seed
    set_seed(args)

    # Get experiment path
    exp_path = get_experiment(args)

    # Load model
    print("Loading model")
    num_classes = len(DATASET_CLASSES[args.dataset])
    model = build_img_classifier(args, num_classes)
    model = model.to(device)
    num_params = count_parameters(model)
    print("Total Parameter: \t%2.1fM" % num_params)

    train(args, model, exp_path, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Config file that overrides default settings when specified
    parser.add_argument("--config", type=argparse.FileType(mode='r'))

    # Choose what to finetune
    parser.add_argument("--model_name", type=str, default=None, help="Specify model name to finetune")
    parser.add_argument("--dataset", type=str, default=None, choices=["rsna_pneumonia", "nih_chest_xray"])

    # To be configured based on hardware/directory
    parser.add_argument("--resume", type=int, default=0, help="input exp number")
    parser.add_argument("--pretrain_path", default=None)
    parser.add_argument("--output_dir", default="./data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=16)

    # Customizable model training settings
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["sgd", "adamw", "adam"])
    parser.add_argument("--freeze_encoder", action="store_true")
    parser.add_argument("--scheduler", type=str, default="cosine")
    parser.add_argument("--data_pct", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--val_steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)

    # Hyperparameter tuning
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--max_grad", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.9)

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.config is not None:
        parser.set_defaults(**yaml.safe_load(args.config))
    args = parser.parse_args() # Reload arguments to override config file values with command line values

    args.phase = "classification"

    main(args)
