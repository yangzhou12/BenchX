import json
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate_fn_map, collate
from torchvision import transforms
from transformers import BertConfig, BertModel
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler

from data_utils.data_radrestruct import RadReStructEval, get_targets_for_split
from evaluation.evaluator_radrestruct import AutoregressiveEvaluator
from evaluation.predict_autoregressive_VQA_radrestruct import predict_autoregressive_VQA
from net.image_encoding import CustomImageEncoder
from net.question_encoding import QuestionEncoderBERT


# implementation of 2D positional encoding from https://github.com/gaopengcuhk/Stable-Pix2Seq
def create_pos_encoding(args):
    temperature = 10000
    hidden_size = args.hidden_size
    max_position_embeddings = args.max_position_embeddings  # args.num_image_tokens + 3 + args.num_question_tokens
    num_pos_feats = hidden_size // 4
    img_mask = torch.ones((1, args.img_feat_size, args.img_feat_size))  # bs, img_tokens
    y_embed = img_mask.cumsum(1, dtype=torch.float32)
    x_embed = img_mask.cumsum(2, dtype=torch.float32)

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t

    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_img = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    pos_img = pos_img.flatten(2)

    # extend pos_img with zeros to match the size of pos_seq
    pos_img = torch.cat(
        (pos_img, torch.zeros((pos_img.shape[0], pos_img.shape[1], max_position_embeddings - pos_img.shape[2]), device=pos_img.device)), dim=2)
    # switch last two dimensions
    pos_img = pos_img.permute(0, 2, 1)

    return pos_img


class MyBertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config, args):
        super().__init__()
        self.args = args
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size // 2)  # other half is reserved for spatial image embeddings
        if args.progressive:
            self.token_type_embeddings = nn.Embedding(4, config.hidden_size)  # image, history Q, history A, current question
        else:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.img_pos_embeddings = create_pos_encoding(self.args)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            past_key_values_length: int = 0,

    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            position_embeddings = torch.cat((self.img_pos_embeddings.to(position_embeddings.device), position_embeddings),
                                            dim=-1)  # add image position embeddings and sequence position embeddings
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class MyBertModel(BertModel):
    """

    Overwrite BERTModel in order to adapt the positional embeddings for images
    """

    def __init__(self, config, add_pooling_layer=True, args=None):
        super().__init__(config)
        self.config = config

        self.embeddings = MyBertEmbeddings(config, args=args)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args

        self.image_encoder = CustomImageEncoder(args)
        self.question_encoder = QuestionEncoderBERT(args)

        self.fusion_config = BertConfig(vocab_size=1, hidden_size=args.hidden_size, num_hidden_layers=args.n_layers,
                                        num_attention_heads=args.heads, intermediate_size=args.hidden_size * 4,
                                        max_position_embeddings=args.max_position_embeddings)

        self.fusion = MyBertModel(config=self.fusion_config, args=args)

        if "vqarad" in args.data_dir:
            self.classifier = nn.Linear(args.hidden_size, args.num_classes)
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(args.classifier_dropout),
                nn.Linear(args.hidden_size, 256),
                nn.ReLU(),
                # nn.BatchNorm1d(256),
                nn.Linear(256, args.num_classes))

    def forward(self, img, input_ids, q_attn_mask, attn_mask, token_type_ids_q=None, mode='train'):

        image_features = self.image_encoder(img, mode=mode)
        text_features = self.question_encoder(input_ids, q_attn_mask)
        cls_tokens = text_features[:, 0:1]

        h = torch.cat((cls_tokens, image_features, text_features[:, 1:]), dim=1)[:, :self.args.max_position_embeddings]
        if self.args.progressive:
            assert token_type_ids_q is not None
            token_type_ids = torch.zeros(h.size(0), h.size(1), dtype=torch.long, device=h.device)
            token_type_ids[:, 0] = 1  # cls token
            token_type_ids[:, 1:image_features.size(1) + 1] = 0  # image features
            token_type_ids[:, image_features.size(1) + 1:] = token_type_ids_q[:, 1:self.args.max_position_embeddings - (
                image_features.size(1))]  # drop CLS token_type_id as added before already and cut unnecessary padding
        else:
            token_type_ids = torch.zeros(h.size(0), h.size(1), dtype=torch.long, device=h.device)
            token_type_ids[:, 0] = 1
            token_type_ids[:, 1:image_features.size(1) + 1] = 0
            token_type_ids[:, image_features.size(1) + 1:] = 1

        out = self.fusion(inputs_embeds=h, attention_mask=attn_mask, token_type_ids=token_type_ids, output_attentions=True)
        h = out['last_hidden_state']
        attentions = out['attentions'][0]
        logits = self.classifier(h.mean(dim=1))

        return logits, attentions


class ModelWrapper(pl.LightningModule):
    def __init__(self, args, train_df=None, val_df=None):
        super(ModelWrapper, self).__init__()
        self.args = args
        self.model = Model(args)
        self.train_df = train_df
        self.val_df = val_df

        if 'radrestruct' in args.data_dir:
            self.test_data_train = get_targets_for_split('train', limit_data=30)
            self.test_data_val = get_targets_for_split('val', limit_data=None)
            self.train_ar_evaluator_vqa = AutoregressiveEvaluator()
            self.val_ar_evaluator_vqa = AutoregressiveEvaluator()

            # handle info dicts in collate_fn
            def collate_dict_fn(batch, *, collate_fn_map):
                return batch

            def custom_collate(batch):
                default_collate_fn_map.update({dict: collate_dict_fn})
                return collate(batch, collate_fn_map=default_collate_fn_map)

            img_tfm = self.model.image_encoder.img_tfm
            norm_tfm = self.model.image_encoder.norm_tfm
            test_tfm = transforms.Compose([img_tfm, norm_tfm]) if norm_tfm is not None else img_tfm
            self.ar_valdataset = RadReStructEval(tfm=test_tfm, mode='val', args=args, limit_data=None)
            self.ar_val_loader_vqa = DataLoader(self.ar_valdataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                                collate_fn=custom_collate)
            self.ar_traindataset = RadReStructEval(tfm=test_tfm, mode='train', args=args, limit_data=30)
            self.ar_train_loader_vqa = DataLoader(self.ar_traindataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
                                                  collate_fn=custom_collate)

            pos_weights_path = 'data/radrestruct/all_pos_weights.json'
            with open(pos_weights_path, 'r') as f:
                self.pos_weight = torch.tensor(json.load(f), dtype=torch.float32)

            self.loss_fn = BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction="none")

        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.train_preds = []
        self.val_preds = []
        self.val_infos = []
        self.val_soft_scores = []
        self.val_gen_preds = []
        self.train_gen_preds = []
        self.train_targets = []
        self.train_soft_scores = []
        self.train_infos = []
        self.train_gen_labels = []
        self.val_targets = []
        self.val_gen_labels = []
        self.train_answer_types = []
        self.val_answer_types = []

        if "radrestruct" in args.data_dir:
            with open('data/radrestruct/answer_options.json', 'r') as f:
                self.answer_options = json.load(f)

    def get_masked_loss(self, loss, mask, targets, path_pos_weight=None):

        if path_pos_weight is not None:
            # for pos predictions multiply loss with path_pos_weight, for negatives with 1 (no change)
            loss = (targets * loss * path_pos_weight) + ((1 - targets) * loss)

        # get mean loss when only considering non-masked options
        masked_loss = loss.masked_select(mask).mean()

        return masked_loss

    def forward(self, img, input_ids, q_attn_mask, attn_mask, token_type_ids_q, mode='train'):
        return self.model(img, input_ids, q_attn_mask, attn_mask, token_type_ids_q, mode=mode)

    def training_step(self, batch, batch_idx, dataset="vqarad"):
        if "vqarad" in self.args.data_dir:
            img, question_token, q_attention_mask, attn_mask, target, answer_type, token_type_ids_q = batch
        else:
            img, question_token, q_attention_mask, attn_mask, target, token_type_ids_q, info, mask = batch

        question_token = question_token.squeeze(1)
        attn_mask = attn_mask.squeeze(1)
        q_attention_mask = q_attention_mask.squeeze(1)

        out, _ = self(img, question_token, q_attention_mask, attn_mask, token_type_ids_q, mode='train')

        logits = out
        if "vqarad" in self.args.data_dir:
            pred = logits.softmax(1).argmax(1).detach()
        else:  # multi-label classification
            pred = (logits.sigmoid().detach() > 0.5).detach().long()
            self.train_soft_scores.append(logits.sigmoid().detach())

        self.train_preds.append(pred)
        self.train_targets.append(target)

        if "vqarad" in self.args.data_dir:
            self.train_answer_types.append(answer_type)
        else:
            self.train_infos.append(info)

        loss = self.loss_fn(logits, target)
        loss = self.get_masked_loss(loss, mask, target, None)  # only use loss of occuring classes

        self.log('Loss/train', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if "vqarad" in self.args.data_dir:
            img, question_token, q_attention_mask, attn_mask, target, answer_type, token_type_ids_q = batch
        else:
            img, question_token, q_attention_mask, attn_mask, target, token_type_ids_q, info, mask = batch

        question_token = question_token.squeeze(1)
        attn_mask = attn_mask.squeeze(1)
        q_attention_mask = q_attention_mask.squeeze(1)

        out, _ = self(img, question_token, q_attention_mask, attn_mask, token_type_ids_q, mode='val')

        logits = out
        if "vqarad" in self.args.data_dir:
            pred = logits.softmax(1).argmax(1).detach()
            self.val_soft_scores.append(logits.softmax(1).detach())
        else:  # multi-label classification
            pred = (logits.sigmoid().detach() > 0.5).detach().long()
            self.val_soft_scores.append(logits.sigmoid().detach())

        self.val_preds.append(pred)
        self.val_targets.append(target)
        if "vqarad" in self.args.data_dir:
            self.val_answer_types.append(answer_type)
        else:
            self.val_infos.append(info)

        if "vqarad" in self.args.data_dir:
            loss = self.loss_fn(logits[target != -1], target[target != -1])
        else:
            loss = self.loss_fn(logits, target.squeeze(0))
            # only use loss of occuring classes
            if "radrestruct" in self.args.data_dir:
                loss = self.get_masked_loss(loss, mask, target, None)
        self.log('Loss/val', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr)
        return optimizer

    def training_epoch_end(self, outputs) -> None:
        preds = torch.cat(self.train_preds).cpu().numpy()
        targets = torch.cat(self.train_targets).cpu().numpy()

        if "vqarad" in self.args.data_dir:
            answer_types = torch.cat(self.train_answer_types).cpu().numpy()
            total_acc = (preds == targets).mean() * 100.
            closed_acc = (preds[answer_types == 0] == targets[answer_types == 0]).mean() * 100.
            open_acc = (preds[answer_types == 1] == targets[answer_types == 1]).mean() * 100.

            self.logger.experiment.add_scalar('Acc/train', total_acc, self.current_epoch)
            self.logger.experiment.add_scalar('ClosedAcc/train', closed_acc, self.current_epoch)
            self.logger.experiment.add_scalar('OpenAcc/train', open_acc, self.current_epoch)

        else:
            if self.current_epoch % 4 == 0:
                # autoregressive evaluation on fixed sub_set of 100 train reports
                preds = predict_autoregressive_VQA(self, self.ar_train_loader_vqa, self.args)
                acc, acc_report, f1, _, _, _ = self.train_ar_evaluator_vqa.evaluate(preds, self.test_data_train)
                self.logger.experiment.add_scalar('Acc/train', acc, self.current_epoch)
                self.logger.experiment.add_scalar('Acc_Report/train', acc_report, self.current_epoch)
                self.logger.experiment.add_scalar('F1/train', f1, self.current_epoch)

        self.train_preds = []
        self.train_targets = []
        self.train_soft_scores = []
        self.train_infos = []
        self.train_answer_types = []
        self.train_gen_labels = []
        self.train_gen_preds = []

    def validation_epoch_end(self, outputs) -> None:
        preds = torch.cat(self.val_preds).cpu().numpy()
        targets = torch.cat(self.val_targets).cpu().numpy()

        if "vqarad" in self.args.data_dir:
            answer_types = torch.cat(self.val_answer_types).cpu().numpy()
            total_acc = (preds == targets).mean() * 100.
            closed_acc = (preds[answer_types == 0] == targets[answer_types == 0]).mean() * 100.
            open_acc = (preds[answer_types == 1] == targets[answer_types == 1]).mean() * 100.

            self.logger.experiment.add_scalar('Acc/val', total_acc, self.current_epoch)
            self.logger.experiment.add_scalar('ClosedAcc/val', closed_acc, self.current_epoch)
            self.logger.experiment.add_scalar('OpenAcc/val', open_acc, self.current_epoch)

            # clean accuracies without samples not occuring in the training set
            total_acc1 = (preds[targets != -1] == targets[targets != -1]).mean() * 100.

            closed_acc1 = (preds[targets != -1][answer_types[targets != -1] == 0] ==
                           targets[targets != -1][answer_types[targets != -1] == 0]).mean() * 100.
            open_acc1 = (preds[targets != -1][answer_types[targets != -1] == 1] ==
                         targets[targets != -1][answer_types[targets != -1] == 1]).mean() * 100.
            # log
            self.log('Acc/val_clean', total_acc1, on_step=False, on_epoch=True, prog_bar=True, logger=False)  # for saving

            self.logger.experiment.add_scalar('Acc/val_clean', total_acc1, self.current_epoch)
            self.logger.experiment.add_scalar('ClosedAcc/val_clean', closed_acc1, self.current_epoch)
            self.logger.experiment.add_scalar('OpenAcc/val_clean', open_acc1, self.current_epoch)

        else:
            if self.current_epoch % 4 == 0:
                # autoregressive evaluation on fixed sub_set of 100 train reports
                preds = predict_autoregressive_VQA(self, self.ar_val_loader_vqa, self.args)
                acc, acc_report, f1, _, _, _ = self.val_ar_evaluator_vqa.evaluate(preds, self.test_data_val)
                self.logger.experiment.add_scalar('Acc/val', acc, self.current_epoch)
                self.logger.experiment.add_scalar('Acc_Report/val', acc_report, self.current_epoch)
                self.logger.experiment.add_scalar('F1/val', f1, self.current_epoch)
                self.log('F1/val', f1, on_step=False, on_epoch=True, prog_bar=True, logger=False)  # for saving
            else:
                self.log('F1/val', 0., on_step=False, on_epoch=True, prog_bar=True, logger=False)  # ignore these epochs in ModelCheckpoint

        self.val_preds = []
        self.val_targets = []
        self.val_soft_scores = []
        self.val_answer_types = []
        self.val_gen_labels = []
        self.val_gen_preds = []
        self.val_infos = []
