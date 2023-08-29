import os
import argparse
import time
import datetime
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from timm.scheduler.cosine_lr import CosineLRScheduler
from pathlib import Path
from tensorboardX import SummaryWriter

import sys
path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))
from datasets.pretraining_dataset import MultimodalPretrainingDataset
from datasets.transforms import DataTransforms
from convirt_module import ConVIRT
from models.convirt.utils import *


def train_one_epoch(model, train_dataloader, optimizer, epoch, warmup_steps, device, lr_scheduler, args, writer):
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.update(loss=1.0)
    metric_logger.update(lr = lr_scheduler._get_lr(epoch)[0])

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps*step_size
    scalar_step = epoch*len(train_dataloader)

    for data_iter_step, batch in enumerate(metric_logger.log_every(train_dataloader, print_freq, header)):

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            imgs = batch["imgs"].to(device)
            caption_ids = batch["caption_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            img_emb, sent_emb = model(imgs, caption_ids, attention_mask, token_type_ids) 

            loss = model.info_nce_loss(img_emb, sent_emb)
            loss.backward()
            optimizer.step()    
            writer.add_scalar('loss/loss', loss, scalar_step)
            scalar_step += 1

            metric_logger.update(loss=loss.item())
            if epoch == 0 and data_iter_step % step_size == 0 and data_iter_step <= warmup_iterations: 
                lr_scheduler.step(data_iter_step // step_size)         
            metric_logger.update(lr=lr_scheduler._get_lr(epoch)[0])
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def main(args):

    # Set up CUDA and GPU
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    torch.set_default_tensor_type('torch.FloatTensor')

    args.deterministic = True
    args.max_epochs = 50

    # define dataloader
    print("Creating dataloader")
    dataset = MultimodalPretrainingDataset(split="train", transform=DataTransforms)
    train_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=None,
            collate_fn=dataset.collate_fn,
            drop_last=True,
        )
    
    # define the model
    model = ConVIRT(img_encoder=args.img_encoder, emb_dim=args.emb_dim, freeze_bert=args.freeze_bert)
    model.to(device)
    print("Model = %s" % str(model))
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, betas=(args.momentum, 0.999), weight_decay=args.weight_decay)
    lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=args.max_epochs,
            cycle_mul=1.0,
            warmup_lr_init=args.learning_rate,
            lr_min=1e-8,
            warmup_t=int(args.max_epochs * 0.4)
        )
    
    # Training steps
    print("Start training")
    start_epoch = 0
    max_epochs = args.max_epochs
    warmup_steps = int(args.max_epochs * 0.4)

    start_time = time.time()
    writer = SummaryWriter(os.path.join(args.output_dir, 'log'))

    for epoch in range(start_epoch, max_epochs):
        if epoch > 0:
            lr_scheduler.step(epoch+warmup_steps)
        train_stats = train_one_epoch(model, train_dataloader, optimizer, epoch, warmup_steps, device, lr_scheduler, args, writer) 

        for k, v in train_stats.items():
            train_loss_epoch = v
        
        writer.add_scalar('loss/train_loss_epoch', float(train_loss_epoch), epoch)
        writer.add_scalar('loss/learning_rate',  lr_scheduler._get_lr(epoch)[0], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch}
        
        # Overwrite and save checkpoint for latest epoch in checkpoint_state.pth
        save_obj = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
        }
        torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_state.pth'))  
        
        with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            f.write(json.dumps(log_stats) + "\n")
        
        # Save every 10th checkpoint
        # if (epoch+1) % 10 == 0 and epoch > 1:
        #     save_obj = {
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'lr_scheduler': lr_scheduler.state_dict(),
        #         'epoch': epoch,
        #     }
        #     torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_'+str(epoch)+'.pth'))         
                
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # trainer args
    parser.add_argument("--img_encoder", type=str, default="resnet_50")
    parser.add_argument("--freeze_bert", action="store_true")
    parser.add_argument("--emb_dim", type=int, default=128, help="128, 256")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--encoder_momentum", type=float, default=0.999)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default='1', help='gpu')
    parser.add_argument("--output_dir", type=str, default='Path/To/Outputdir')
    args = parser.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.cuda.current_device()
    torch.cuda._initialized = True
    torch.manual_seed(args.seed)

    main(args)