import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

from unifier.blocks.vision import *
from unifier.blocks.pytorch import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoder,
)
from unifier.blocks.custom.r2gen import EncoderDecoder
from unifier.blocks.pytorch import PositionalEncoding
from unifier.blocks.losses import R2GenLMLoss
from unifier.models.utils import get_n_params
from unifier.blocks.losses import *
from unifier.blocks.custom.refers.transformer import REFERSViT



def evaluation(models, config, dl, **kwargs):
    tokenizer = dl.dataset.tokenizer
    # max_len = dl.dataset.max_words

    losses = np.zeros((len(dl), len(models)))
    eval_gts, eval_res = [], []

    with torch.no_grad():
        for num_batch, batch in enumerate(tqdm(dl, total=len(dl))):
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            for num_model, model in enumerate(models):
                result = model(batch["images"], from_training=False)
                losses[num_batch][num_model] = result["loss"].cpu().item()
                reports = tokenizer.batch_decode(result["output"].cpu().numpy())
                ground_truths = tokenizer.batch_decode(
                    batch["reports_ids"][:, 1:].cpu().numpy()
                )
                eval_res.extend(reports)
                eval_gts.extend(ground_truths)
            
    eval_res = {i: [re] for i, re in enumerate(eval_res)}
    eval_gts = {i: [gt] for i, gt in enumerate(eval_gts)}

    loss = np.mean(losses)

    return {
        "loss": loss,
        "refs": eval_gts,
        "hyps": eval_res,
    }


class R2Gen(nn.Module):
    def __init__(self, cnn, transformer, loss, **kwargs):
        super(R2Gen, self).__init__()

        # Build visual extractor
        cnn_func = cnn.pop("proto")
        self.vit_pool = cnn.pop("vit_pool", "")
        self.cnn = eval(cnn_func)(**cnn)
        if 'output_layer' in cnn and cnn.output_layer == "layer3": # Hard code
            self.avg_fnt = torch.nn.AvgPool2d(kernel_size=14, stride=1, padding=0)
        else:
            self.avg_fnt = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)

        # Build transformer
        self.encoder_decoder = EncoderDecoder(transformer)

        # Logits layer
        self.logit = nn.Linear(transformer.d_model, transformer.vocab_size + 1)

        # Evaluation
        loss_func = loss.pop("proto")
        self.loss_func = eval(loss_func)(**loss).cuda()

        print(self.loss_func)

        self.eval_func = evaluation

    def forward_vision(self, images):
        # Set no_permute for visual extractor to get direct output of final layer
        patch_feats = self.cnn(images) # [B, hidden_size, H, W]

        if patch_feats.dim() == 3: # ViT
            if self.vit_pool == "token":
                avg_feats = patch_feats[:, 0] # class token (Timm implementation)
            elif self.vit_pool == "avg":
                avg_feats = patch_feats[:, 1:, :].mean(1)
            else:
                raise RuntimeError("Invilid Pooling Type")
        elif patch_feats.dim() == 4: #  CNN
            avg_feats = self.avg_fnt(patch_feats).squeeze().reshape(-1, patch_feats.size(1)) # [B, hidden_size]
            batch_size, feat_size, _, _ = patch_feats.shape
            patch_feats = patch_feats.reshape(batch_size, feat_size, -1).permute(0, 2, 1)
        else:
            raise RuntimeError("Dimensions are wrong")

        return patch_feats, avg_feats

    def forward(
        self,
        images,
        reports_ids=None,
        reports_masks=None,
        from_training=True,
        test=False,
        iteration=None,
        epoch=None,
        **kwargs
    ):
        images = images.cuda()
        if from_training:
            reports_ids = reports_ids.cuda()
            reports_masks = reports_masks.cuda()

        att_feats_0, fc_feats_0 = self.forward_vision(images[:, 0])
        att_feats_1, fc_feats_1 = self.forward_vision(images[:, 1])
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)

        if from_training:
            output = self.encoder_decoder(
                fc_feats, att_feats, reports_ids, mode="forward"
            )
        else:
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode="sample")

        loss = torch.tensor(0.0)
        if from_training:
            # classification loss
            # loss = self.loss_func(output.view(output.size(0) * output.size(1), -1), reports_ids[:, 1:].reshape(-1), **kwargs)
            # R2Gen loss
            loss = self.loss_func(output, reports_ids[:, 1:], reports_masks[:, 1:], **kwargs)

        return {"loss": loss, "output": output}

    def __repr__(self):
        s = super().__repr__() + "\n"
        s += "{}\n".format(get_n_params(self))
        return s
