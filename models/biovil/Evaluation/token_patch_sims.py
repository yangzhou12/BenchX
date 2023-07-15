import argparse
import sys
sys.path.insert(0, '/n/data2/hms/XXXX-9/XXXX-4/XXXX-1/TIER_Regularized_CLIP/Modeling/')
import torch
import CLIP_Embedding
import MedDataHelpers
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

print("CUDA Available: " + str(torch.cuda.is_available()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main(args):
    resultDir = args.results_dir
    dat = MedDataHelpers.getDatasets(source = 'mimic_cxr', subset = ['test'], frontal=True, filters=['impression'])
    dl = MedDataHelpers.getLoaders(dat, shuffle=False)
    dl = dl['test']

    unreg_model = CLIP_Embedding.getCLIPModel(args.je_model_path_unreg).to(device)
    clip_model = CLIP_Embedding.getCLIPModel(args.je_model_path).to(device)
    soft = nn.Softmax(dim=1)

    with torch.no_grad():
        for i, samples in enumerate(dl):
            text = samples['texts'] #corresponding tex
            ims = samples['images'] #first im in list
            if i == 0:
                print(text[0])

            if i == 0:
                img = ims[0] #N CH DIM DIM
                img[:, 0, :, :] = (img[:, 0, :, :] * .229) + .485
                img[:, 1, :, :] = (img[:, 1, :, :] * .224) + .456
                img[:, 2, :, :] = (img[:, 2, :, :] * .225) + .406
                img = img.permute(0, 2, 3, 1).squeeze()

            unreg_im = unreg_model.get_im_embeddings(ims, only_patches=True)
            unreg_embeds = unreg_model.get_text_embeddings(text, only_words=True)
            unreg_cross_weights = unreg_model.get_cross_weights(unreg_im, unreg_embeds)[0]


            token_im = clip_model.get_im_embeddings(ims, only_patches = True)
            token_embeds = clip_model.get_text_embeddings(text, only_words=True)
            cross_weights_text = clip_model.get_cross_weights(token_im, token_embeds)[0]

            cwt = cross_weights_text.shape
            print(cwt)
            maxlen = 256
            if cwt[1] < maxlen:
                cross_weights_text = torch.cat([cross_weights_text, -2 * torch.ones(cwt[0], maxlen-cwt[1], cwt[2]).to(device)], dim=1)
                cross_weights_text[cross_weights_text == -2] = float('nan')

                unreg_cross_weights = torch.cat([unreg_cross_weights, -2 * torch.ones(cwt[0], maxlen-cwt[1], cwt[2]).to(device)], dim=1)
                unreg_cross_weights[unreg_cross_weights == -2] = float('nan')

            if i == 0:
                allattns = cross_weights_text
                allunreg = unreg_cross_weights
            else:
                if i == 10:
                    break
                allattns = torch.cat([allattns, cross_weights_text], dim=0) #D T P
                allunreg = torch.cat([allunreg, unreg_cross_weights])


        lol = allattns[:, 0, :].squeeze().cpu() #raw similarities
        lolunreg = allunreg[:, 0, :].squeeze().cpu()

        lolnew = lol/(lol.sum(dim=1)[:, None].repeat(1, 49)) #relative normalized
        lolunregnew = lolunreg / (lolunreg.sum(dim=1)[:, None].repeat(1, 49))

        lol, indices = torch.sort(lol, dim=1, descending=True) #T P
        lolunreg, indicesunreg = torch.sort(lolunreg, dim=1, descending=True)
        descending_attn_means = np.nanmean(lol, axis=0)
        descending_attn_stds = np.nanstd(lol, axis=0)
        descending_attn_means_unreg = np.nanmean(lolunreg, axis=0)
        descending_attn_stds_unreg = np.nanstd(lolunreg, axis=0)
        fig, ax = plt.subplots(1, 1, figsize=(6,6))
        ax.errorbar(np.arange(descending_attn_means.shape[0]), descending_attn_means, yerr = descending_attn_stds, fmt='.', linestyle='', ecolor='b', label="Regularized")
        ax.errorbar(np.arange(descending_attn_means_unreg.shape[0]), descending_attn_means_unreg, yerr=descending_attn_stds_unreg, fmt='.',
                    linestyle='', ecolor='r', label="Unregularized")
        ax.set_title("Rank-ordered patch similarities", size=18)
        ax.set_xlabel("Rank-ordered patch #", size=16)
        ax.set_ylabel("Patch similarities to CLS text embedding", size=16)
        ax.legend(prop={'size': 16})
        plt.savefig(resultDir + args.je_model_path[-5:-1] + 'raw_sim.png', bbox_inches='tight')

        lolnew, indices = torch.sort(lolnew, dim=1, descending=True)  # T P
        lolunregnew, indicesunreg = torch.sort(lolunregnew, dim=1, descending=True)
        descending_attn_means = np.nanmean(lolnew, axis=0)
        descending_attn_stds = np.nanstd(lolnew, axis=0)
        descending_attn_means_unreg = np.nanmean(lolunregnew, axis=0)
        descending_attn_stds_unreg = np.nanstd(lolunregnew, axis=0)
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.errorbar(np.arange(descending_attn_means.shape[0]), descending_attn_means, yerr=descending_attn_stds,
                    fmt='.', linestyle='', ecolor='b', label="Regularized")
        ax.errorbar(np.arange(descending_attn_means_unreg.shape[0]), descending_attn_means_unreg,
                    yerr=descending_attn_stds_unreg, fmt='.',
                    linestyle='', ecolor='r', label="Unregularized")

        ax.set_title("Rank-ordered relative patch importance", size=18)
        ax.set_xlabel("Rank-ordered patch #", size=16)
        ax.set_ylabel("Similarity/Total Similarity", size=16)
        ax.legend(prop={'size': 16})
        plt.savefig(resultDir + args.je_model_path[-5:-1] + 'relative_sim.png', bbox_inches='tight')


if __name__ == '__main__': #7, 8, 9 = norm, patch, both
    parser = argparse.ArgumentParser()
    parser.add_argument('--je_model_path_unreg', type=str, default='/n/data2/hms/XXXX-9/XXXX-4/XXXX-1/Med_ImageText_Embedding/models/clip_regularized/exp3/')
    parser.add_argument('--je_model_path', type=str, default='/n/data2/hms/XXXX-9/XXXX-4/XXXX-1/Med_ImageText_Embedding/models/clip_regularized/exp2/', help='path for saving trained models')
    parser.add_argument('--embed_size', type=int, default=128, help='dimension of word embedding vectors')
    parser.add_argument('--batch_size', type=int, default=32) #32 normally
    parser.add_argument('--results_dir',type=str, default='/n/data2/hms/XXXX-9/XXXX-4/XXXX-1/Med_ImageText_Embedding/results/attn/')
    parser.add_argument('--dat_dir', type=str, default='/n/data2/hms/XXXX-9/XXXX-4/XXXX-1/Med_ImageText_Embedding/data/')
    args = parser.parse_args()
    print(args)
    main(args)