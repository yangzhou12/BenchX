import os
import re
import math
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

import sys
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from models.builders import *
from datasets.classification_dataset import *
from evaluation.ssl_classifier import *
from datasets.transforms import DataTransforms
from datasets.transforms import MRMTransforms
from evaluation.backbones import cnn_backbones, vit_backbones



#########################################
#          Experiment Setup             #      
#########################################

def getExperiment(args):
    # TODO: Implement resume function
    # if args.resume and args.resume > 0:
    #     fp =  os.path.join(args.output_dir, 'exp'+str(args.resume))
    #     if os.path.exists(fp):
    #         return fp
    # else:
    #     raise Exception("Experiment doesn't exist, cannot resume exp " + fp)

    if not os.listdir(os.path.join(args.output_dir)):
        print("No models exist, creating directory")
        fp = os.path.join(args.output_dir, 'exp1')
    else:
        all_files = os.listdir(os.path.join(args.output_dir))
        je_exps = [exp for exp in all_files if 'exp' in exp]
        num = [int(re.search('\d+', exp).group(0)) for exp in je_exps]
        highest_ind = np.argmax(np.array(num))
        highest = num[highest_ind]
        highest = highest + 1
        fp = os.path.join(args.output_dir, 'exp'+str(highest))
    return fp

def load_training_setting(args, optimizer, lr_scheduler):
    checkpoint_model = torch.load(args.pretrain_path, map_location='cpu')
    optimizer.load_state_dict(checkpoint_model['optimizer'])
    lr_scheduler.load_state_dict(checkpoint_model['lr_scheduler'])
    start_epoch = checkpoint_model['epoch']
    return start_epoch, optimizer, lr_scheduler


######################################################
#            Load Datasets & Class Names             #      
######################################################

# Store dataset mappings of value format (dataset_class, num_classes)
DATASETS = {'rsna_pneumonia': (RSNAImageDataset, 1), 'nih_chest_xray': (NIHChestXRay14, 14)} 

def build_dataloaders(args):
    if args.dataset in DATASETS:
        dataset_class, num_classes = DATASETS[args.dataset]
    else:
        raise RuntimeError(
            "Please specify a dataset.\n" +
            "Run --help to see datasets available for downstream task."
        )

    train_dataset = dataset_class('train', MRMTransforms, 
                                         args.phase, args.data_pct, args.image_res)
    train_dataloader = DataLoader(
            train_dataset,
            batch_size = args.batch_size,
            num_workers = args.num_workers,
            pin_memory=True,
            sampler=None,
            shuffle=True,
            collate_fn=None,
            drop_last=True,
        )
    
    val_dataset = dataset_class('valid', MRMTransforms,
                                       args.phase, args.data_pct, args.image_res)
    val_dataloader = DataLoader(
            val_dataset,
            batch_size= args.batch_size,
            num_workers= args.num_workers,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=False,
        )
    
    test_dataset = dataset_class('test', MRMTransforms, 
                                        args.phase, args.data_pct, args.image_res)
    test_dataloader = DataLoader(
            test_dataset,
            batch_size= args.batch_size,
            num_workers= args.num_workers,
            pin_memory=True,
            sampler=None,
            shuffle=False,
            collate_fn=None,
            drop_last=False,
        )
    
    return train_dataloader, val_dataloader, test_dataloader, num_classes

def get_classes(args):
    if args.dataset not in DATASETS:
        raise RuntimeError(
            f"Dataset {args.dataset} does not exist."
        )
    
    if args.dataset == "nih_chest_xray":
        return ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
        'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    elif args.dataset == "rsna_pneumonia":
        return ['Pneumonia']
    

#########################################
#            SSL Classifier             #      
#########################################

# Model-specific function
def build_ssl_classifier(args, num_classes):
    if args.model_name == "biovil":
        model = build_biovil_model(args, eval=True)
        img_backbone = model.cnn.encoder
        ssl_classifier = SSLClassifier(backbone=img_backbone, num_classes=num_classes)
    elif args.model_name == "mrm":
        ssl_classifier = build_mrm_classifier(args, num_classes)
    elif args.model_name == "gloria":
        img_backbone = build_gloria_encoder(args)
        ssl_classifier = SSLClassifier(backbone=img_backbone, num_classes=num_classes)
    elif args.model_name == "convirt":
        img_backbone = build_convirt_encoder(args)
        ssl_classifier = SSLClassifier(backbone=img_backbone, num_classes=num_classes)
    else:
        raise RuntimeError("Model not found")
    
    return ssl_classifier

# General function
# TODO: Add contract - make contract for checkpoint to store model weights at 'model' key
def build_img_classifier(args, num_classes):
    if args.model_type == 'cnn':
        img_backbone = cnn_backbones.__dict__[args.model_name](pretrained=True)
    elif args.model_type == 'vit':
        img_backbone = vit_backbones.__dict__[args.model_name]()
    else:
        raise RuntimeError("Model type not found")
    
    # Wrapper model class with linear classifier head
    ssl_classifier = SSLClassifier(backbone=img_backbone, num_classes=num_classes) 
    
    if args.pretrain_path: # load pre-trained model
        checkpoint_model = torch.load(args.pretrain_path, map_location='cpu')['model']
        msg = ssl_classifier.load_state_dict(checkpoint_model, strict=False)
        print(msg)
    
    return ssl_classifier


#############################################
#           MRM Linear Scheduler            #      
#############################################

class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))

class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))