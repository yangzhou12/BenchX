import torch
import os
import numpy as np
import re
import sys
import shutil

sys.path.insert(0, '/n/data2/hms/XXXX-9/XXXX-4/XXXX-1/TIER_Regularized_CLIP/Modeling/')
import utils
import MedDataHelpers
print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

use_zeroshot_val = True
N = 5
cutoff = 1000
path = '/n/data2/hms/XXXX-9/XXXX-4/XXXX-1/Med_ImageText_Embedding/models/clip_regularized_findings/exp3/'
all_files = os.listdir(path)
all_models = np.array([f for f in all_files if 'je_model' in f])
all_mod_numbers = np.array([int(re.search(r'\d+', model).group()) for model in all_models])
mod_order = np.argsort(all_mod_numbers)
all_models = all_models[mod_order]
all_mod_numbers = all_mod_numbers[mod_order]
all_models = all_models[all_mod_numbers <= cutoff]
all_mod_numbers = all_mod_numbers[all_mod_numbers <= cutoff]

val_losses = []
for i, model in enumerate(all_models):
    print(all_mod_numbers[i])
    checkpoint = torch.load(path + model, map_location=device)
    if use_zeroshot_val:
        dat = MedDataHelpers.getDatasets(source = 'c', subset=['test'])
        DLs = MedDataHelpers.getLoaders(dat, shuffle=False)
        DL = DLs['test']
        val_loss = utils.getZeroShotAcc(checkpoint, DL)
        val_losses.append(val_loss)
    else:
        val_losses.append(-1 * checkpoint['val_loss'])

print(str(N), " to Save: ")
val_losses = np.array(val_losses)
topNindex = np.argsort(val_losses)[-N:][::-1] #top N acc / neg loss, in descending order
val_losses_to_save = val_losses[topNindex]
model_numbers_to_save = all_mod_numbers[topNindex]
models_to_save = all_models[topNindex]
for i, model in enumerate(models_to_save):
    print(model_numbers_to_save[i], val_losses_to_save[i])
    shutil.copyfile(path + model, path + "best_model_" + str(i) + ".pt")