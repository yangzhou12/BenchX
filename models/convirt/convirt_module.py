import torch
import torch.nn.functional as F
from torch import nn

import sys
from pathlib import Path
path_root = Path(__file__).parents[0]
sys.path.append(str(path_root))
from encoders import *


class ConVIRT(nn.Module):
    def __init__(self, img_encoder="resnet_50", emb_dim=128, freeze_bert=False):
        super().__init__()
        self.img_encoder = ImageEncoder(model_name=img_encoder, output_dim=emb_dim)
        self.text_encoder = BertEncoder(output_dim=emb_dim, freeze_bert=freeze_bert)

    def forward(self, img, caption_ids, attention_mask, token_type_ids):
        img_feat, _ = self.img_encoder(img)
        img_emb = self.img_encoder.global_embed(img_feat)
        img_emb = F.normalize(img_emb, dim=1)

        sent_feat, _, _, _ = self.text_encoder(
            caption_ids, attention_mask, token_type_ids)
        sent_emb = self.text_encoder.global_embed(sent_feat)
        sent_emb = F.normalize(sent_emb, dim=1)

        return img_emb, sent_emb
    
    @staticmethod
    def info_nce_loss(out_1, out_2, temperature=0.07):
        bz = out_1.size(0)
        labels = torch.arange(bz).type_as(out_1).long()

        scores = out_1.mm(out_2.t())
        scores /= temperature
        scores1 = scores.transpose(0, 1)
        loss0 = nn.CrossEntropyLoss()(scores, labels)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)

        return loss0 + loss1