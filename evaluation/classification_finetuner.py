import os
import torch
import torch.nn.functional as F
import argparse
import time
import ruamel.yaml as yaml
import datetime
import json
import numpy as np
from pathlib import Path
from tensorboardX import SummaryWriter
from timm.scheduler.cosine_lr import CosineLRScheduler
from sklearn.metrics import roc_auc_score, precision_recall_curve, accuracy_score

import sys
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from evaluation.utils import *
from train_utils import *



def train(model, data_loader, optimizer, criterion, epoch, warmup_steps, device, scheduler, args, writer):
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.update(loss=1.0)
    metric_logger.update(lr = scheduler._get_lr(epoch)[0])

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size 
    scalar_step = epoch*len(data_loader)

    for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = sample['image']
        label = sample['label'].float().to(device) #batch_size, num_class
        input_image = image.to(device, non_blocking=True)  

        optimizer.zero_grad()
        pred_class = model(input_image) #batch_size, num_class

        loss = criterion(pred_class, label)
        loss.backward()
        optimizer.step()    
        writer.add_scalar('loss/loss', loss, scalar_step)
        scalar_step += 1

        metric_logger.update(loss=loss.item())
        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations: 
            scheduler.step(i // step_size)         
        metric_logger.update(lr=scheduler._get_lr(epoch)[0])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def valid(model, data_loader, criterion, epoch, device, writer):
    model.eval()
    val_scalar_step = epoch*len(data_loader)
    val_losses = []
    for i, sample in enumerate(data_loader):
        image = sample['image']
        label = sample['label'].float().to(device)
        input_image = image.to(device,non_blocking=True)  
        with torch.no_grad():
            pred_class = model(input_image)
            val_loss = criterion(pred_class,label)
            val_losses.append(val_loss.item())
            writer.add_scalar('val_loss/loss', val_loss, val_scalar_step)
            val_scalar_step += 1
    avg_val_loss = np.array(val_losses).mean()
    return avg_val_loss


def test(model, data_loader, device, num_classes):
    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    print("Start testing")
    model.eval()
    for i, sample in enumerate(data_loader):
        image = sample['image']
        label = sample['label'].float().to(device)
        gt = torch.cat((gt, label), 0)
        input_image = image.to(device,non_blocking=True)  
        with torch.no_grad():
            pred_class = model(input_image)
            pred_class = F.sigmoid(pred_class)
            pred = torch.cat((pred, pred_class), 0)
    
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

    AUROCs = compute_AUCs(gt, pred, num_classes)
    AUROC_avg = np.array(AUROCs).mean()
    gt_np = gt[:, 0].cpu().numpy()
    pred_np = pred[:, 0].cpu().numpy()            
    precision, recall, thresholds = precision_recall_curve(gt_np, pred_np)
    numerator = 2 * recall * precision
    denom = recall + precision
    f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
    max_f1 = np.max(f1_scores)
    max_f1_thresh = thresholds[np.argmax(f1_scores)]
    print('The max f1 is',max_f1)
    print('The accuracy is', accuracy_score(gt_np, pred_np>max_f1_thresh)) 
    print('The average AUROC is {AUROC_avg:.3f}'.format(AUROC_avg=AUROC_avg))
    for i in range(num_classes):
        print('The AUROC of Pneumonia is {}'.format(AUROCs[i]))
    return AUROC_avg


def main(args):

    # Set up CUDA and GPU
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    torch.set_default_tensor_type('torch.FloatTensor')

    # Get experiment path
    exp_path = getExperiment(args)

    # Load specified dataset
    print("Creating dataset")
    train_dataloader, val_dataloader, test_dataloader, num_classes = build_dataloaders('rsna_pneumonia', args)
    
    # Load model
    print("Loading model")
    model = build_ssl_classifier(args, num_classes) 
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = CosineLRScheduler(optimizer, t_initial=args.num_epochs, warmup_t=args.warmup_epochs, 
                                     lr_min=1e-5, cycle_decay=1., warmup_lr_init=1e-5) #Setting for MedKLIP
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    # Training steps
    print("Start training")
    start_epoch = 0
    max_epochs = args.num_epochs
    warmup_steps = args.warmup_epochs

    start_time = time.time()
    best_val_loss = 10000
    writer = SummaryWriter(os.path.join(exp_path, 'log'))

    for epoch in range(start_epoch, max_epochs):
        if epoch > 0:
            lr_scheduler.step(epoch+warmup_steps)
        train_stats = train(model, train_dataloader, optimizer, criterion, epoch, warmup_steps, device, lr_scheduler, args, writer) 

        for k, v in train_stats.items():
            train_loss_epoch = v
        
        writer.add_scalar('loss/train_loss_epoch', float(train_loss_epoch), epoch)
        writer.add_scalar('loss/learning_rate',  lr_scheduler._get_lr(epoch)[0], epoch)

        val_loss = valid(model, val_dataloader, criterion, epoch, device, writer)
        writer.add_scalar('loss/val_loss_epoch', val_loss, epoch)
 
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch, 'val_loss': val_loss.item()
                    }                     
        save_obj = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
        }
        torch.save(save_obj, os.path.join(exp_path, 'checkpoint_state.pth'))  
        
        with open(os.path.join(exp_path, "log.txt"),"a") as f:
            f.write(json.dumps(log_stats) + "\n")
        
        if val_loss < best_val_loss:
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(exp_path, 'best_valid.pth'))  
            best_val_loss = val_loss
            args.model_path = os.path.join(exp_path, 'best_valid.pth')
            test_auc = test(model, test_dataloader, device, num_classes)
            with open(os.path.join(exp_path, "log.txt"), "a") as f:
                f.write('The average AUROC is {AUROC_avg:.4f}'.format(AUROC_avg=test_auc) + "\n")
        
        if (epoch+1) % 10 == 0 and epoch > 1:
            save_obj = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(exp_path, 'checkpoint_'+str(epoch)+'.pth'))         
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Customizable model training parameters
    parser.add_argument('--dataset', type=str, default='', help='options: rsna_pneumonia')
    parser.add_argument('--model_name', type=str, default='', help='options: biovil')
    parser.add_argument('--data_pct', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--image_res', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)

    # To be configured based on hardware/directory
    parser.add_argument('--checkpoint', default='') 
    parser.add_argument('--model_path', default='') 
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
    torch.manual_seed(args.seed)

    # Set downstream task to classification
    args.phase = 'classification'

    main(args)