import os
import torch
import argparse
import shutil
import time
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
from models.builders import *



def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    
    def forward(self, pred, target):
            smooth = 1.0

            pred = torch.sigmoid(pred)

            p_flat = pred.view(-1)
            t_flat = target.view(-1)
            intersection = (p_flat * t_flat).sum()
            return (2.0 * intersection + smooth) / (p_flat.sum() + t_flat.sum() + smooth)


def get_dice(probability, truth, threshold=0.5):
    batch_size = len(truth)
    with torch.no_grad():
        probability = probability.view(batch_size, -1)
        truth = truth.view(batch_size, -1)
        assert probability.shape == truth.shape

        p = (probability > threshold).float()
        t = (truth > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])

    return torch.mean(dice).detach().item()


def valid(model, data_loader, criterion, device, writer, global_step):
    model.eval()
    val_losses = []
    for i, sample in enumerate(tqdm(data_loader)):
        image = sample['image']
        label = sample['mask'].float().to(device)
        input_image = image.to(device, non_blocking=True)  
        with torch.no_grad():
            pred_class = model(input_image).squeeze(dim=1)
            val_loss = criterion(pred_class, label)
            val_losses.append(val_loss.item())
    avg_val_loss = np.array(val_losses).mean()
    writer.add_scalar('valid/loss', avg_val_loss, global_step)
    return avg_val_loss


# Calculate mIoU
def test(model, data_loader, criterion, device, writer, global_step):
    return


def main(args):
    # Set up CUDA and GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    args.n_gpu = torch.cuda.device_count()
    torch.set_default_tensor_type('torch.FloatTensor')
    
    # Set seed
    set_seed(args)

    # Get experiment path
    exp_path = get_experiment(args)

    # Load specified dataset
    print("Creating dataset")
    train_dataloader, val_dataloader, test_dataloader = get_ft_dataloaders(args) 
    classes = DATASET_CLASSES[args.dataset]
    
    # Load model
    print("Loading model")
    model = smp.Unet("resnet50", encoder_weights=None, activation=None)

    if args.pretrain_path:
        ckpt = torch.load(args.ckpt_path)
        ckpt_dict = dict()
        for k, v in ckpt["state_dict"].items():
            if k.startwith("img_encoder.model"):
                new_k = ".".join(k.split(".")[2:])
                ckpt_dict[new_k] = v
        
        ckpt_dict["fc.bias"] = None
        ckpt_dict["fc.weight"] = None

        args.seg_model.encoder.load_state_dict(ckpt_dict)

        # Freeze encoder
        for param in args.seg_model.encoder.parameters():
            param.requires_grad = False

    model = model.to(device)
    num_params = count_parameters(model)
    print("Total Parameter: \t%2.1fM" % num_params) 

    # Prepare loss
    criterion = DiceLoss()

    # Training steps
    print("Start training")
    start_time = time.time()
    best_val_loss = 10000
    writer = SummaryWriter(os.path.join(exp_path, 'log'))

    # Save copy of run_finetuning.sh file in exp folder
    shutil.copyfile('run_finetuning_seg.sh', os.path.join(exp_path, 'run_finetuning_seg.sh'))

    # Train by batch
    t_total = args.num_steps
    global_step = 0

    global_step, optimizer, lr_scheduler = load_training_setup(args, exp_path, model)

    while True:
        model.train()
        
        epoch_iterator = tqdm(train_dataloader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True)

        for i, sample in enumerate(epoch_iterator):
            image = sample['image']
            label = sample['mask'].float().to(device) #batch_size, num_class
            input_image = image.to(device, non_blocking=True)  

            optimizer.zero_grad()
            logit = model(input_image) #batch_size, num_class
            logit = logit.squeeze(dim=1)

            loss = criterion(logit, label)
            loss.backward()

            prob = torch.sigmoid(logit)
            dice = get_dice(prob, label)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad) # Tackle exploding gradients
            optimizer.step()
            lr_scheduler.step(global_step)
            global_step += 1

            epoch_iterator.set_description(
                "Training (%d / %d Steps) (loss=%2.5f) (dice=%2.5f)" % (global_step, t_total, loss, dice)
            )
            writer.add_scalar('train/loss', loss, global_step)
            writer.add_scalar('train/lr', lr_scheduler._get_lr(global_step)[0], global_step)

            # Validate every T steps, as specified by val_step
            if args.val_steps >= 0:
                val_step = args.val_steps
            else:
                val_step = len(train_dataloader)

            if global_step % val_step == 0:

                print("\n" + f"Start validating at step {global_step}")
                val_loss = valid(model, val_dataloader, criterion, device, writer, global_step)
                
                # Log training and validation stats at validation step
                log_stats = {'loss': loss.item(), 'val_loss': val_loss.item(), 'lr': lr_scheduler._get_lr(global_step)[0],
                             'step': global_step}
                with open(os.path.join(exp_path, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                save_obj = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'step': global_step
                }

                # Save most recent tested checkpoint at each specified validation interval
                torch.save(save_obj, os.path.join(exp_path, 'checkpoint_state.pth'))  

                # Save and test model with best validation score
                # if val_loss < best_val_loss:
                #     torch.save(save_obj, os.path.join(exp_path, 'best_valid.pth'))  
                #     best_val_loss = val_loss
                #     args.model_path = os.path.join(exp_path, 'best_valid.pth')
                #     test_auc = test(args, model, test_dataloader, device, classes)
                    
                #     with open(os.path.join(exp_path, "log.txt"), "a") as f:
                #         f.write('The average AUROC is {AUROC_avg:.4f}'.format(AUROC_avg=test_auc) + "\n") 

            if global_step % t_total == 0:
                break

        if global_step % t_total == 0:
            break



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Customizable model training settings
    parser.add_argument('--model_name', type=str, default='', help='model name')
    parser.add_argument('--dataset', default="siim_acr_pneumothorax")
    parser.add_argument('--optimizer', type=str, default='', choices=['sgd', 'adamw'])
    parser.add_argument('--scheduler', type=str, default='', choices=['cosine'])
    parser.add_argument('--data_pct', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--warmup_steps', type=int, default=20)
    parser.add_argument('--val_steps', type=int, default=-1, help="specify a number, else default value is length of dataset")
    parser.add_argument('--seed', type=int, default=42)
    
    # Hyperparameter tuning
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--max_grad', type=float, default=1.0)
    parser.add_argument('--momentum', type=float, default=0.9)

    # To be configured based on hardware/directory
    parser.add_argument('--resume', type=int, default=0, help='input exp number') 
    parser.add_argument('--pretrain_path', default='Path/To/checkpoint.pth')
    parser.add_argument('--output_dir', default='Path/To/Outputdir')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--num_workers', type=int, default=4)
    
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.current_device()
    torch.cuda._initialized = True

    main(args)