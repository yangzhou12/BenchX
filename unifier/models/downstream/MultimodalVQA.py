import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from transformers import AutoConfig

from unifier.blocks.vision import *
from unifier.blocks.huggingface.encoder.encoder_model import EncoderModel
from unifier.blocks.losses import *
from unifier.blocks.bert.BertCrossLayer import BertCrossLayer
from unifier.blocks.classifier.evaluation import evaluation
from unifier.models.utils import get_n_params


def init_weights(module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class MultimodalVQA(nn.Module):
    def __init__(self, encoder, cnn, fusion, adapter, classifier, loss, **kwargs):
        super(MultimodalVQA, self).__init__()

        # == Begin: 0. Initialize BERT config for co-attention ==
        bert_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=encoder.proto,
            vocab_size=fusion["vocab_size"],
            hidden_size=fusion["hidden_size"],
            num_hidden_layers=fusion["num_layers"],
            num_attention_heads=fusion["num_heads"],
            intermediate_size=fusion["hidden_size"] * fusion["mlp_ratio"],
            max_position_embeddings=fusion["max_text_len"],
            hidden_dropout_prob=fusion["drop_rate"],
            attention_probs_dropout_prob=fusion["drop_rate"],
        )

        # == Begin: 1. Build Models ==
        cnn_func = cnn.pop("proto")

        self.vision_encoder = eval(cnn_func)(**cnn)
        self.language_encoder = EncoderModel(encoder)

        self.multi_modal_language_proj = nn.Linear(
            fusion.pop("input_text_embed_size"), fusion.get("hidden_size")
        )
        self.multi_modal_language_proj.apply(init_weights)
        self.multi_modal_vision_proj = nn.Linear(
            fusion.pop("input_image_embed_size"), fusion.get("hidden_size")
        )
        self.multi_modal_vision_proj.apply(init_weights)

        # Adapter
        self.adapter = nn.Sequential(
            nn.Linear(adapter.pop("input_size"), adapter.pop("output_size")),
            torch.nn.LayerNorm(
                fusion.get("hidden_size"),
                eps=adapter.pop("eps"),
            ),
        )
        self.adapter.apply(init_weights)

        self.modality_type_embeddings = nn.Embedding(2, fusion.get("hidden_size"))
        self.modality_type_embeddings.apply(init_weights)

        self.multi_modal_vision_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(fusion.get("num_top_layer"))]
        )
        self.multi_modal_vision_layers.apply(init_weights)
        self.multi_modal_language_layers = nn.ModuleList(
            [BertCrossLayer(bert_config) for _ in range(fusion.get("num_top_layer"))]
        )
        self.multi_modal_language_layers.apply(init_weights)

        self.multi_modal_vision_pooler = Pooler(fusion.get("hidden_size"))
        self.multi_modal_vision_pooler.apply(init_weights)
        self.multi_modal_language_pooler = Pooler(fusion.get("hidden_size"))
        self.multi_modal_language_pooler.apply(init_weights)

        # == Begin: 2. Build VQA Head ==
        hs = fusion.get("hidden_size")
        self.vqa_label_size = classifier.get("num_classes")
        self.vqa_head = nn.Sequential(
            nn.Linear(hs * 2, hs * 2),
            nn.LayerNorm(hs * 2),
            nn.GELU(),
            nn.Linear(hs * 2, self.vqa_label_size),
        )
        self.vqa_head.apply(init_weights)

        # == Begin: 3. Evaluation ==
        loss_func = loss.pop("proto")
        self.loss_func = eval(loss_func)(**loss).cuda()

        print(self.loss_func)

        self.eval_func = evaluation

    def infer(self, batch, image_token_type_idx=1):
        # == Begin: Fetch the inputs ==
        img = batch["images"].cuda()
        text_ids = batch["text_ids"].cuda()
        text_masks = batch["text_masks"].cuda()
        # == End  : Fetch the inputs ==

        # == Begin: Text Encoding ==
        unimodal_text_feats = self.language_encoder.encoder.embeddings(
            input_ids=text_ids
        )
        extended_text_masks = self.language_encoder.encoder.get_extended_attention_mask(
            text_masks, text_masks.size()
        ).cuda()

        for layer in self.language_encoder.encoder.encoder.layer:
            unimodal_text_feats = layer(unimodal_text_feats, extended_text_masks)[0]
        unimodal_text_feats = self.multi_modal_language_proj(unimodal_text_feats)
        # == End  : Text Encoding ==

        # == Begin: Image Encoding ==
        unimodal_image_feats = self.vision_encoder.forward(img)

        # Adapter to account for ResNet-50 implementations
        unimodal_image_feats = self.adapter(unimodal_image_feats)

        unimodal_image_feats = self.multi_modal_vision_proj(unimodal_image_feats)
        image_masks = torch.ones(
            (unimodal_image_feats.size(0), unimodal_image_feats.size(1)),
            dtype=torch.long,
        ).cuda()  # only take first two dimensions
        extended_image_masks = (
            self.language_encoder.encoder.get_extended_attention_mask(
                image_masks, image_masks.size()
            ).cuda()
        )
        # == End  : Image Encoding ==

        # == Begin: Assign Type Embeddings ==
        unimodal_text_feats, unimodal_image_feats = (
            unimodal_text_feats
            + self.modality_type_embeddings(torch.zeros_like(text_masks)),
            unimodal_image_feats
            + self.modality_type_embeddings(
                torch.full_like(image_masks, image_token_type_idx)
            ),
        )
        # == End  : Assign Type Embeddings ==

        # == Begin: Multi-Modal Fusion ==
        x, y = unimodal_text_feats, unimodal_image_feats
        for layer_idx, (text_layer, image_layer) in enumerate(
            zip(self.multi_modal_language_layers, self.multi_modal_vision_layers)
        ):
            # == Begin: Co-Attention ==
            x1 = text_layer(
                x, y, extended_text_masks, extended_image_masks, output_attentions=True
            )
            y1 = image_layer(
                y, x, extended_image_masks, extended_text_masks, output_attentions=True
            )
            x, y = x1[0], y1[0]
            # == End: Co-Attention ==
        # == End  : Multi-Modal Fusion ==

        # == Begin: == Output Multi-Modal Features ==
        multimodal_text_cls_feats = self.multi_modal_language_pooler(x)
        multimodal_image_cls_feats = self.multi_modal_vision_pooler(y)
        multimodal_cls_feats = torch.cat(
            [multimodal_text_cls_feats, multimodal_image_cls_feats], dim=-1
        )
        # == End  : == Output Multi-Modal Features ==

        return multimodal_cls_feats

    def forward(
        self,
        images,
        texts=None,
        labels=None,
        from_training=True,
        iteration=None,
        epoch=None,
        **kwargs
    ):
        batch = {
            "images": images,
            "text_ids": texts["text_ids"],
            "text_masks": texts["text_masks"],
        }

        multimodal_cls_feats = self.infer(batch)
        vqa_logits = self.vqa_head(multimodal_cls_feats)

        vqa_targets = torch.zeros(len(vqa_logits), self.vqa_label_size)

        vqa_labels = labels
        for i, _label in enumerate(vqa_labels):
            vqa_targets[i, _label] = 1.0

        loss = torch.tensor(0.0)
        if from_training:
            loss = (
                self.loss_func(vqa_logits, vqa_targets.cuda(), **kwargs)
                * vqa_targets.shape[1]
            )

        return {"loss": loss, "output": vqa_logits}

    def __repr__(self):
        s = super().__repr__() + "\n"
        s += "{}\n".format(get_n_params(self))
        return s
