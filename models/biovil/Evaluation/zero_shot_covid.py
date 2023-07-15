import argparse
import sys
sys.path.insert(0, '/n/data2/hms/XXXX-9/XXXX-4/XXXX-1/TIER_Regularized_CLIP/Modeling/')
import torch
import CLIP_Embedding
import MedDataHelpers
import utils
from torchmetrics import AUROC
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import metrics

print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_best_thresh(fprs, tprs, thresholds):
    dist = np.sqrt(np.square(fprs) + np.square(1-tprs))
    best_thresh_ind = np.argmin(dist)
    return thresholds[best_thresh_ind]

def main(args):
    heads1 = np.array(['covid19', 'Pneumonia'])
    heads2 = np.array(['covid19', 'Pneumonia', 'No Finding'])
    if args.subset == 'a' or args.subset == 'all':
        subset = ['all']
    elif args.subset == 't' or args.subset == 'test':
        subset = ['test']
    clip_models = CLIP_Embedding.getCLIPModel(args.je_model_path, num_models=1)
    clip_models = [clip_models]
    filters = MedDataHelpers.getFilters(args.je_model_path)
    modname = args.je_model_path[-5:-1]

    dat = MedDataHelpers.getDatasets(source=args.sr, subset=subset, synthetic=False, filters = filters, heads = heads1) #Real
    DLs = MedDataHelpers.getLoaders(dat, args, shuffle=False)
    DL = DLs[subset[0]]

    aucs, aucs_synth, aucs_adv, tprs, fprs, thresholds, accs = {}, {}, {}, {}, {}, {}, {}
    auroc = AUROC(pos_label=1)

    for k, clip_model in enumerate(clip_models):
        test_preds, test_targets = utils.get_all_preds(DL, clip_model,similarity=True, heads=heads2)
        test_preds = test_preds[0].cpu()
        test_targets = test_targets.cpu()
        test_preds = test_preds[:, :2]
        test_preds[:, 0] = test_preds[:, 0] - test_preds[:, 1]
        if k == 0:
            tot_preds = test_preds
        else:
            tot_preds += test_preds


    test_preds = tot_preds


    for i, h in enumerate(heads1):
        fprs[h], tprs[h], thresholds[h] = metrics.roc_curve(test_targets[:, i].int().detach().numpy(), test_preds[:, i].detach().numpy())
        best_thresh = get_best_thresh(fprs[h], tprs[h], thresholds[h])
        accs[h] = metrics.accuracy_score(test_targets[:, i].int().detach().numpy(), test_preds[:, i].detach().numpy() > best_thresh)
        print(h, "acc", accs[h])
        aucs[h] = np.round(auroc(test_preds[:, i], test_targets[:, i].int()).item(), 5)
    aucs['Total'] = np.round(np.mean(np.array([aucs[h] for h in heads1])), 5)
    print("Normal")
    print("Total AUC avg: ", aucs['Total'])
    for i, h in enumerate(heads1):
        print(h, aucs[h])





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--je_model_path', type=str, default='/n/data2/hms/XXXX-9/XXXX-4/XXXX-1/Med_ImageText_Embedding/models/clip_regularized/exp3/', help='path for saving trained models')
    parser.add_argument('--sr', type=str, default='co') #c, co
    parser.add_argument('--subset', type=str, default='all')
    parser.add_argument('--embed_size', type=int, default=128, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=32) #32 normally
    parser.add_argument('--results_dir',type=str, default='/n/data2/hms/XXXX-9/XXXX-4/XXXX-1/Med_ImageText_Embedding/results/zeroshot/')
    parser.add_argument('--dat_dir', type=str, default='/n/data2/hms/XXXX-9/XXXX-4/XXXX-1/Med_ImageText_Embedding/data/')
    args = parser.parse_args()
    print(args)
    main(args)