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
import pickle
import pandas as pd

print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["TOKENIZERS_PARALLELISM"] = "false"



def getOpt(preds,labels, thresh):
    thresh = thresh.astype(float)
    bestMCC, bestF1 = -1, 0
    bestF1thresh, bestMCCthresh = -1, -1
    for t in thresh:
        finalpreds = preds > t
        MCC = metrics.matthews_corrcoef(labels, finalpreds)
        F1 = metrics.f1_score(labels, finalpreds)
        if MCC > bestMCC:
            bestMCC = MCC
            bestMCCthresh = t
        if F1 > bestF1:
            bestF1 = F1
            bestF1thresh = t
    return bestMCCthresh, bestF1thresh

def getScores(targs, tpreds, optF1Thresh, optMCCThresh):
    return metrics.matthews_corrcoef(targs, tpreds>optMCCThresh), metrics.f1_score(targs, tpreds >optF1Thresh)
def getRadScores(radpreds, targs):
    return metrics.matthews_corrcoef(targs, radpreds), metrics.f1_score(targs, radpreds)


def main(args):
    rerun = False
    heads = np.array(['Cardiomegaly', 'Edema', 'Consolidation', 'Atelectasis', 'Pleural Effusion'])
    if args.subset == 'a' or args.subset == 'all':
        subset = ['all']
    elif args.subset == 't' or args.subset == 'test':
        subset = ['test']

    squash = False
    aucs, aucs_synth, aucs_adv, tprs, fprs, thresholds, optMCCThresh, optF1Thresh, MCC, F1 = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}
    rad1MCC, rad1F1, rad2MCC, rad2F1, rad3MCC, rad3F1 = {}, {}, {}, {}, {}, {}
    auroc = AUROC(pos_label=1)
    filters = MedDataHelpers.getFilters(args.je_model_path)
    modname = os.path.basename(args.je_model_path)
    dat = MedDataHelpers.getDatasets(source='c', subset=['test'], synthetic=False, filters=filters,
                                     heads=heads)  # Real
    DLs = MedDataHelpers.getLoaders(dat, args, shuffle=False)
    DLval = DLs[subset[0]]

    dat = MedDataHelpers.getDatasets(source=args.sr, subset=subset, synthetic=False, filters=filters,
                                     heads=heads)  # Real
    DLs = MedDataHelpers.getLoaders(dat, args, shuffle=False)
    DLtest = DLs[subset[0]]

    #clip_models = [CLIP_Embedding.getCLIPModel(args.je_model_path)]
    clip_models = CLIP_Embedding.getCLIPModel(args.je_model_path, num_models = 5)
    tot_preds = 0
    for cnum, clip_model in enumerate(clip_models):
        if os.path.isfile(args.je_model_path + 'valpredictions/' + args.sr + str(cnum) + ('squash' if squash else '') + 'preds.pickle') and not rerun:
            with open(args.je_model_path + 'valpredictions/' + args.sr + str(cnum) + ('squash' if squash else '')+ 'preds.pickle', 'rb') as handle:
                test_preds, test_targets = pickle.load(handle)
                print("Loading val preds")
        else:
            test_preds, test_targets = utils.get_all_preds(DLval, clip_model,similarity=True, heads=heads, convirt=True, normalization=True, squash_studies=squash) #N c
            test_preds = test_preds[0].cpu().detach().numpy()
            test_targets = test_targets.cpu().int().detach().numpy()
            with open(args.je_model_path + 'valpredictions/' + args.sr + str(cnum) +('squash' if squash else '')+ 'preds.pickle', 'wb') as handle:
                pickle.dump([test_preds, test_targets], handle, protocol=pickle.HIGHEST_PROTOCOL)

        if cnum == 0:
            tot_preds = test_preds
        else:
            tot_preds = tot_preds + test_preds

    tot_preds = tot_preds / (cnum + 1)
    #Val AUC
    for i, h in enumerate(heads):
        targs = test_targets[:, i]
        tpreds = tot_preds[:, i]
        print(targs, tpreds)
        fprs[h], tprs[h], thresholds[h] = metrics.roc_curve(targs, tpreds)
        aucs[h] = np.round(metrics.auc(fprs[h], tprs[h]), 5)
        optMCCThresh[h], optF1Thresh[h] = getOpt(tpreds, targs, thresholds[h])
    aucs['Total'] = np.round(np.mean(np.array([aucs[h] for h in heads])), 5)
    print("Val AUCs:")
    print("Total AUC avg: ", aucs['Total'])
    for i, h in enumerate(heads):
        print(h, aucs[h])

    #Test

    for cnum, clip_model in enumerate(clip_models):
        if os.path.isfile(args.je_model_path + 'predictions/' + args.sr + str(cnum) + ('squash' if squash else '') + 'radpreds.pickle') and not rerun:
            with open(args.je_model_path + 'predictions/' + args.sr + str(cnum) +('squash' if squash else '') +  'radpreds.pickle', 'rb') as handle:
                test_preds, test_targets, rad1preds, rad2preds, rad3preds = pickle.load(handle)
        else:
            test_preds, test_targets, rad1preds, rad2preds, rad3preds = utils.get_all_preds(DLtest, clip_model,similarity=True, heads=heads, convirt=True, normalization=True, squash_studies=squash, getrad=True) #N c
            test_preds = test_preds[0].cpu().detach().numpy()
            test_targets = test_targets.cpu().int().detach().numpy()
            with open(args.je_model_path + 'predictions/' + args.sr + str(cnum) + ('squash' if squash else '') + 'radpreds.pickle', 'wb') as handle:
                pickle.dump([test_preds, test_targets, rad1preds, rad2preds, rad3preds], handle, protocol=pickle.HIGHEST_PROTOCOL)

        if cnum == 0:
            tot_preds = test_preds
        else:
            tot_preds = tot_preds + test_preds

    tot_preds = tot_preds / (cnum + 1)

    #Test AUc and scores
    for i, h in enumerate(heads):
        targs = test_targets[:, i]
        tpreds = tot_preds[:, i]
        fprs[h], tprs[h], thresholds[h] = metrics.roc_curve(targs, tpreds)
        MCC[h], F1[h] = getScores(targs, tpreds, optF1Thresh[h], optMCCThresh[h])
        rad1MCC[h], rad1F1[h] = getRadScores(rad1preds[:, i], targs)
        rad2MCC[h], rad2F1[h] = getRadScores(rad2preds[:, i], targs)
        rad3MCC[h], rad3F1[h] = getRadScores(rad3preds[:, i], targs)

        aucs[h] = np.round(metrics.auc(fprs[h], tprs[h]), 5)
    aucs['Total'] = np.round(np.mean(np.array([aucs[h] for h in heads])), 5)
    print("Test AUCs:")
    print("Total AUC avg: ", aucs['Total'])
    for i, h in enumerate(heads):
        print(h, aucs[h])



    MCC['Total'] = np.round(np.mean(np.array([MCC[h] for h in heads])), 5)
    F1['Total'] = np.round(np.mean(np.array([F1[h] for h in heads])), 5)

    with open(args.je_model_path + 'predictions/metricsresults.pickle', 'wb') as handle:
        pickle.dump([aucs, MCC, F1], handle, protocol=pickle.HIGHEST_PROTOCOL)

    rad1MCC['Total'] = np.round(np.mean(np.array([rad1MCC[h] for h in heads])), 5)
    rad2MCC['Total'] = np.round(np.mean(np.array([rad2MCC[h] for h in heads])), 5)
    rad3MCC['Total'] = np.round(np.mean(np.array([rad3MCC[h] for h in heads])), 5)
    rad1F1['Total'] = np.round(np.mean(np.array([rad1F1[h] for h in heads])), 5)
    rad2F1['Total'] = np.round(np.mean(np.array([rad2F1[h] for h in heads])), 5)
    rad3F1['Total'] = np.round(np.mean(np.array([rad3F1[h] for h in heads])), 5)
    radTotalMCC = np.round(np.mean(np.array([rad['Total'] for rad in [rad1MCC, rad2MCC, rad3MCC]])), 5)
    radTotalF1 = np.round(np.mean(np.array([rad['Total'] for rad in [rad1F1, rad2F1, rad3F1]])), 5)

    with open(args.je_model_path + 'predictions/radiologistresults.pickle', 'wb') as handle:
        pickle.dump([rad1MCC, rad2MCC, rad3MCC, rad1F1, rad2F1, rad3F1], handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("MCC Scores:")
    print("My Avg: ", MCC['Total'])
    print("Rad Avg: ", radTotalMCC)
    print("Rad 1 Avg: ", rad1MCC['Total'])
    print("Rad 2 Avg: ", rad2MCC['Total'])
    print("Rad 3 Avg: ", rad3MCC['Total'])

    print("F1 Scores:")
    print("My Avg: ", F1['Total'])
    print("Rad Avg: ", radTotalF1)
    print("Rad 1 Avg: ", rad1F1['Total'])
    print("Rad 2 Avg: ", rad2F1['Total'])
    print("Rad 3 Avg: ", rad3F1['Total'])

    for i, h in enumerate(heads):
        print(h)
        print("MCC Scores:")
        print("My: ", MCC[h])
        print("Rad: ", np.round(np.mean(np.array([rad[h] for rad in [rad1MCC, rad2MCC, rad3MCC]])), 5))
        print("Rad 1: ", rad1MCC[h])
        print("Rad 2: ", rad2MCC[h])
        print("Rad 3: ", rad3MCC[h])

        print("F1 Scores:")
        print("My: ", F1[h])
        print("Rad: ", np.round(np.mean(np.array([rad[h] for rad in [rad1F1, rad2F1, rad3F1]])), 5))
        print("Rad 1: ", rad1F1[h])
        print("Rad 2: ", rad2F1[h])
        print("Rad 3: ", rad3F1[h])






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--je_model_path', type=str, default='/n/data2/hms/XXXX-9/XXXX-4/XXXX-1/Med_ImageText_Embedding/models/clip_regularized/exp2/', help='path for saving trained models')
    parser.add_argument('--sr', type=str, default='chextest') #c, co
    parser.add_argument('--subset', type=str, default='test')
    parser.add_argument('--embed_size', type=int, default=128, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=32) #32 normally
    parser.add_argument('--results_dir',type=str, default='/n/data2/hms/XXXX-9/XXXX-4/XXXX-1/Med_ImageText_Embedding/results/zeroshot/')
    parser.add_argument('--dat_dir', type=str, default='/n/data2/hms/XXXX-9/XXXX-4/XXXX-1/Med_ImageText_Embedding/data/')
    args = parser.parse_args()
    print(args)
    main(args)