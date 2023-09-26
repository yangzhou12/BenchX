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
from timm.models.vision_transformer import VisionTransformer

from datasets.dataloader import get_ft_dataloaders
from evaluation.utils import *
from evaluation.segmentation.segmentation_loss import *
from evaluation.segmentation.metrics import *
from evaluation.segmentation.transformer_seg import SETRModel
from utils.constants import *


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def build_img_segmenter(args):
    """
    Return an image segmenter based on model argument.
    """

    if args.base_model == "resnet50":
        model = smp.Unet("resnet50", encoder_weights=None, activation=None)
        load_encoder_from_checkpoint(args, model.encoder)

        # Freeze encoder
        if args.freeze_encoder:
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
        encoder = vit_base_patch16(num_classes=0, global_pool="", class_token=True) # class_token=False for MRM
        model.encoder_2d.bert_model = encoder

        load_encoder_from_checkpoint(args, model.encoder_2d.bert_model)

        # Freeze encoder
        if args.freeze_encoder:
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


def valid(model, data_loader, criterion, num_classes, device, writer, global_step):
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()
    
    model.eval()
    val_losses = []
    for i, sample in enumerate(data_loader):
        image = sample["image"]
        label = sample["mask"].float().to(device)
        gt = torch.cat((gt, label), 0)
        input_image = image.to(device, non_blocking=True)
        with torch.no_grad():
            logit = model(input_image).squeeze(dim=1)
            prob = torch.sigmoid(logit)
            pred = torch.cat((pred, prob), 0)
            val_loss = criterion(logit, label)
            val_losses.append(val_loss.item())
    
    avg_val_loss = np.array(val_losses).mean()
    ret_metrics = compute_metrics(pred, gt, num_classes, "medDice")
    writer.add_scalar("valid/loss", avg_val_loss, global_step)
    writer.add_scalar("valid/dice", ret_metrics["medDice"], global_step)
    return avg_val_loss, ret_metrics


def test(args, model, num_classes, writer, data_loader, device, global_step):
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

    tqdm.write("Test results at step {curr_step}:".format(curr_step=global_step))
    for metric, value in ret_metrics.items():
        tqdm.write(
            "The average {metric_name} is {metric_avg:.5f}".format(
                metric_name=metric, metric_avg=value
            )
        )
    writer.add_scalar('test/dice', ret_metrics['medDice'], global_step)

    return ret_metrics


def train(args, model, exp_path, device):
    # Load specified dataset
    print("Creating dataset")
    train_dataloader, val_dataloader, test_dataloader = get_ft_dataloaders(args)

    # Prepare training loss
    criterion = MixedLoss()
    num_classes = 2  # binary segmentation task

    # Training steps
    print("Start training")
    start_time = time.time()
    best_dice = 0
    writer = SummaryWriter(os.path.join(exp_path, "log"))

    # Save copy of run_finetuning.sh file in exp folder
    shutil.copyfile(
        "run_finetuning_seg.sh", os.path.join(exp_path, "run_finetuning_seg.sh")
    )

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
            label = sample["mask"].float().to(device)  # batch_size, num_class
            input_image = image.to(device, non_blocking=True)

            optimizer.zero_grad()
            logit = model(input_image)  # batch_size, num_class
            logit = logit.squeeze(dim=1)

            loss = criterion(logit, label)
            loss.backward()

            prob = torch.sigmoid(logit)
            ret_medDice = compute_metrics(prob, label, num_classes, metrics=["medDice"])
            dice = ret_medDice["medDice"]

            # if i == 0:
            #     img = sample["image"][0].cpu().numpy()
            #     mask = sample["mask"][0].cpu().numpy()
            #     mask = np.stack([mask, mask, mask])

            #     layered = 0.6 * mask + 0.4 * img
            #     img = img.transpose((1, 2, 0))
            #     mask = mask.transpose((1, 2, 0))
            #     layered = layered.transpose((1, 2, 0))

            optimizer.step()
            lr_scheduler.step(global_step)
            global_step += 1

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f) (dice=%2.5f)"
                % (global_step, t_total, loss, dice)
            )
            writer.add_scalar("train/loss", loss, global_step)
            writer.add_scalar(
                "train/lr", lr_scheduler._get_lr(global_step)[0], global_step
            )
            writer.add_scalar("train/dice", dice, global_step)

            if args.val_steps >= 0:
                val_step = args.val_steps
            else:
                val_step = len(train_dataloader)

            if global_step % val_step == 0 or global_step % t_total == 0:
                epoch_iterator.set_description(
                    "Validating... (%d / %d Steps) (loss=%2.5f) (dice=%2.5f)"
                    % (global_step, t_total, loss, dice)
                )

                val_loss, val_metrics = valid(
                    model, val_dataloader, criterion, num_classes, device, writer, global_step
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
                val_dice = val_metrics["medDice"]
                if val_dice > best_dice:
                    torch.save(save_obj, os.path.join(exp_path, "best_valid.pth"))
                    best_dice = val_dice

                    epoch_iterator.set_description(
                        "Testing... (%d / %d Steps) (val_loss=%2.5f) (val_dice=%2.5f)"
                        % (global_step, t_total, val_loss, val_dice)
                    )
                    ret_metrics = test(
                        args, model, num_classes, writer, test_dataloader, device, global_step
                    )

                    with open(os.path.join(exp_path, "log.txt"), "a") as f:
                        f.write("Metrics: " + str(ret_metrics) + "\n")

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
    model = build_img_segmenter(args)
    model = model.to(device)
    num_params = count_parameters(model)
    print("Total Parameter: \t%2.1fM" % num_params)

    train(args, model, exp_path, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Mandatory finetuning parameters to specify
    parser.add_argument("--model_name", type=str, required=True, help="model name")
    parser.add_argument("--base_model", type=str, required=True, choices=["vit", "resnet50"])
    parser.add_argument("--dataset", type=str, required=True, choices=["siim_acr_pneumothorax", "rsna_segmentation"])
    
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
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--val_steps", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)

    # Hyperparameter tuning
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--momentum", type=float, default=0.9)

    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    args.phase = "segmentation"

    main(args)