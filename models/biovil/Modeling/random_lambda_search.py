import time
t = time.time()
import argparse
import torch
print("CUDA Available: " + str(torch.cuda.is_available()))
from utils import *
import sys
import random
sys.path.insert(0, '/n/data2/hms/XXXX-9/XXXX-4/XXXX-1/Med_ImageText_Embedding/src/evaluate/')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

elapsed = time.time() - t
print("Start (time = " + str(elapsed) + ")")


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    best_val_acc = 0
    best_val_patch = -100
    best_val_word = -100
    filters=['impression']
    mimic_dat = MedDataHelpers.getDatasets(source='m', subset=['trainmed'], augs=1, filters=filters)
    dls = MedDataHelpers.getLoaders(mimic_dat, args, shuffle=False)
    train_data_loader_mimic = dls['trainmed']

    dat = MedDataHelpers.getDatasets(source='c', subset=['test'])
    DLs = MedDataHelpers.getLoaders(dat, shuffle=False)
    chexdl = DLs['test']

    for i in range(30):
        if i == 0:
            random_patch, random_words = 0, 0
        else:
            random_patch, random_words = random.random()/2, random.random()/2
        clip_model = CLIP_Embedding.MedCLIP(eval=False).to(device)
        params = list(clip_model.parameters())
        optimizer = torch.optim.Adam(params, lr=args.learning_rate, weight_decay=0.000001)
        for epoch in range(1):
            clip_model.cnn.train(True)
            clip_model.transformer.train(True)
            train_loss, train_losses = train(train_data_loader_mimic, clip_model, args, epoch, optimizer, lam_patch=random_patch, lam_words = random_words)
            clip_model.cnn.train(False)
            clip_model.transformer.train(False)
            val_acc = getZeroShotAcc(clip_model, chexdl, usecheckpoint=False)
            print("zero-shot Acc:",val_acc,"patch:", random_patch,"words:", random_words)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_patch = random_patch
                best_val_word = random_words
            print("Best zero-shot Acc:",best_val_acc,"patch:", best_val_patch,"words:", best_val_word)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #model information
    parser.add_argument('--model_path', type=str, default='/n/data2/hms/XXXX-9/XXXX-4/XXXX-1/Med_ImageText_Embedding/models/', help='path for saving trained models')
    parser.add_argument('--model', type=str, default='clip_regularized')

    #Training parameters
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32) #32 vs 16
    parser.add_argument('--learning_rate', type=float, default=.0001) #.0001
    args = parser.parse_args()
    print(args)
    main(args)

