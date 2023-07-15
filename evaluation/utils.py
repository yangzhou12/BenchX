import os
import re
import numpy as np
from torch.utils.data import DataLoader

import sys
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))
from models.builders import *
from datasets.classification_dataset import *
from evaluation.ssl_classifier import *
from datasets.transforms import DataTransforms



#########################################
#          Experiment Setup             #      
#########################################

def getExperiment(args):
    # TODO: Implement resume
    # if args.resume and args.resume > 0:
    #     fp =  os.path.join(output_dir, 'exp'+str(args.resume))
    #     if os.path.exists(fp):
    #         return fp
    #     else:
    #         raise Exception("Experiment doesn't exist, cannot resume exp " + fp)

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


#########################################
#            Load Datasets              #      
#########################################

# Store dataset mappings of value format (dataset_class, num_classes)
DATASETS = {'rsna_pneumonia': (RSNAImageDataset, 1)} 

def build_dataloaders(dataset_name, args):
    if dataset_name in DATASETS:
        dataset_class, num_classes = DATASETS[dataset_name]
    else:
        raise RuntimeError(
            "Please specify a dataset.\n" +
            "Run --help to see datasets available for downstream task."
        )

    train_dataset = dataset_class('train', DataTransforms, 
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
    
    val_dataset = dataset_class('valid', DataTransforms,
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
    
    test_dataset = dataset_class('test', DataTransforms, 
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


#########################################
#            SSL Classifier             #      
#########################################

def build_ssl_classifier(args, num_classes):
    img_backbone = get_img_backbone(args)
    ssl_classifier = SSLClassifier(backbone=img_backbone, num_classes=num_classes)
    return ssl_classifier

def get_img_backbone(args):
    if args.model_name == "biovil":
        model = build_biovil_model(args, eval=True)
        return model.cnn.encoder
    else:
        raise RuntimeError("Model not found")