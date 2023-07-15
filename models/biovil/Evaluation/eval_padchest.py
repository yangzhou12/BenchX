import argparse
import sys
sys.path.insert(0, '/n/data2/hms/XXXX-9/XXXX-4/XXXX-1/TIER_Regularized_CLIP/Modeling/')
import torch
import CLIP_Embedding
import MedDataHelpers
import utils
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn import metrics
import pickle
rerun = False
print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_to_eval = '/n/data2/hms/XXXX-9/XXXX-4/XXXX-1/Med_ImageText_Embedding/models/clip_regularized/exp1/'
num_models = 2
sub = 'all'
picklename = model_to_eval + 'padchest_preds_' + str(num_models) + '_' + sub + '.pickle'
aucsname = model_to_eval + 'padchest_aucs_' + str(num_models) + '_' + sub + '.pickle'

if not os.path.isfile(picklename) or rerun:
    clip_models = CLIP_Embedding.getCLIPModel(modelpath=model_to_eval, num_models=num_models, eval=True)
    if num_models == 1:
        clip_models = [clip_models]
    dat = MedDataHelpers.getDatasets(source = 'padchest', subset = [sub], filters = [])
    print(dat.__len__())
    dl = MedDataHelpers.getLoaders(dat, zeroworkers=True, shuffle=False)
    alldat = dl[sub]
    all_preds, all_targs, names = utils.getPadPredictions(alldat, clip_models) #gets all predictions for each target in logit form
    for k, n in enumerate(names):
        print(n, all_targs[:, k].sum())
    predinfo = {'all_preds': all_preds, 'all_targs':all_targs, 'names':names}
    with open(picklename, 'wb') as handle:
        pickle.dump(predinfo, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open(picklename, 'rb') as handle:
        predinfo = pickle.load(handle)
        all_preds = predinfo['all_preds']
        all_targs = predinfo['all_targs']
        names = predinfo['names']

print(all_preds.shape, all_targs.shape, len(names))
aucs, fprs, tprs, thresholds = {}, {}, {}, {}
for i, h in enumerate(names):
    targs = all_targs[:, i]
    tpreds = all_preds[:, i]
    fprs[h], tprs[h], thresholds[h] = metrics.roc_curve(targs, tpreds)
    aucs[h] = np.round(metrics.auc(fprs[h], tprs[h]), 5)

all = np.array([aucs[h] for h in names])
print(all)
aucs['Avg'] = np.mean(all)
print("Avg", np.round(aucs['Avg'], 3))
for i, h in enumerate(names):
    print(h, np.round(aucs[h], 3))

with open(aucsname, 'wb') as handle:
    pickle.dump([aucs, names], handle, protocol=pickle.HIGHEST_PROTOCOL)


