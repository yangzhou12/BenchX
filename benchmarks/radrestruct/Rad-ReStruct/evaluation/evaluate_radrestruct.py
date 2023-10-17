import argparse
import json
import os
import warnings

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import collate, default_collate_fn_map
from torchvision import transforms

from data_utils.data_radrestruct import RadReStructEval, get_targets_for_split
from evaluation.evaluator_radrestruct import AutoregressiveEvaluator
from net.model import ModelWrapper
from evaluation.predict_autoregressive_VQA_radrestruct import predict_autoregressive_VQA

warnings.simplefilter("ignore", UserWarning)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Finetune on RadReStruct")

    parser.add_argument('--run_name', type=str, required=False, default="debug", help="run name for wandb")
    parser.add_argument('--data_dir', type=str, required=False, default="data/radrestruct", help="path for data")
    parser.add_argument('--model_dir', type=str, required=False, default="",
                        help="path to load weights")
    parser.add_argument('--save_dir', type=str, required=False, default="checkpoints_radrestruct", help="path to save weights")
    parser.add_argument('--question_type', type=str, required=False, default=None, help="choose specific category if you want")
    parser.add_argument('--use_pretrained', action='store_true', default=False, help="use pretrained weights or not")
    parser.add_argument('--mixed_precision', action='store_true', default=False, help="use mixed precision or not")

    parser.add_argument('--bert_model', type=str, required=False, default="zzxslp/RadBERT-RoBERTa-4m", help="pretrained question encoder weights")

    parser.add_argument('--progressive', action='store_true', default=False, help="use progressive answering of questions")
    parser.add_argument('--match_instances', action='store_true', default=False, help="do optimal instance matching")
    parser.add_argument('--aug_history', action='store_true', default=False, help="do history augmentation")

    parser.add_argument('--seed', type=int, required=False, default=42, help="set seed for reproducibility")
    parser.add_argument('--num_workers', type=int, required=False, default=4, help="number of workers")
    parser.add_argument('--epochs', type=int, required=False, default=100, help="num epochs to train")
    parser.add_argument('--classifier_dropout', type=float, required=False, default=0.0, help="how often should image be dropped")

    parser.add_argument('--max_position_embeddings', type=int, required=False, default=12, help="max length of sequence")
    parser.add_argument('--max_answer_len', type=int, required=False, default=29, help="padding length for free-text answers")
    parser.add_argument('--batch_size', type=int, required=False, default=16, help="batch size")
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help="learning rate'")
    # parser.add_argument('--weight_decay', type = float, required = False, default = 1e-2, help = " weight decay for gradients")

    parser.add_argument('--hidden_dropout_prob', type=float, required=False, default=0.3, help="hidden dropout probability")

    parser.add_argument('--img_feat_size', type=int, required=False, default=14, help="dimension of last pooling layer of img encoder")
    parser.add_argument('--num_question_tokens', type=int, required=False, default=20, help="number of tokens for question")
    parser.add_argument('--hidden_size', type=int, required=False, default=768, help="hidden size")
    parser.add_argument('--vocab_size', type=int, required=False, default=30522, help="vocab size")
    parser.add_argument('--type_vocab_size', type=int, required=False, default=2, help="type vocab size")
    parser.add_argument('--heads', type=int, required=False, default=16, help="heads")
    parser.add_argument('--n_layers', type=int, required=False, default=1, help="num of layers")
    parser.add_argument('--acc_grad_batches', type=int, required=False, default=None, help="how many batches to accumulate gradients")

    # add arguments for custom checkpoints
    parser.add_argument("--base_model", type=str, default="", choices=["", "vit", "resnet50"]) #by default, use EfficientNet

    args = parser.parse_args()

    # same as vqarad progressive
    args.num_image_tokens = args.img_feat_size ** 2
    args.max_position_embeddings = 458
    args.hidden_size_img_enc = args.hidden_size
    args.num_question_tokens = 458 - 3 - args.num_image_tokens

    # create directory for saving params
    if not os.path.exists(f'{args.save_dir}/{args.run_name}'):
        os.makedirs(f'{args.save_dir}/{args.run_name}')
    with open(os.path.join(args.save_dir, f'{args.run_name}/commandline_args.txt'), 'w+') as f:
        json.dump(args.__dict__, f, indent=2)

    pl.seed_everything(args.seed, workers=True)

    args.num_classes = 96

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = ModelWrapper(args)

    # use torchinfo to see model architecture and trainable parameters
    from torchinfo import summary

    summary(model)
    model.to(device)

    if args.use_pretrained:
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(args.model_dir, map_location=torch.device('cpu'))['state_dict'])
        assert len(missing_keys) == 0
        assert len(unexpected_keys) == 0

    img_tfm = model.model.image_encoder.img_tfm
    norm_tfm = model.model.image_encoder.norm_tfm
    resize_size = model.model.image_encoder.resize_size

    test_tfm = transforms.Compose([img_tfm, norm_tfm]) if norm_tfm is not None else img_tfm

    valdataset = RadReStructEval(tfm=test_tfm, mode='test', args=args)


    # handle info dicts in collate_fn
    def collate_dict_fn(batch, *, collate_fn_map):
        return batch


    def custom_collate(batch):
        default_collate_fn_map.update({dict: collate_dict_fn})
        return collate(batch, collate_fn_map=default_collate_fn_map)


    valloader = DataLoader(valdataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate)

    logger = pl.loggers.TensorBoardLogger('runs_radrestruct', name=args.run_name, version=0)

    preds = predict_autoregressive_VQA(model, valloader, args)

    evaluator = AutoregressiveEvaluator()

    targets = get_targets_for_split('test')

    acc, acc_report, f1, precision, recall, detailed_metrics = evaluator.evaluate(preds, targets[:len(preds)])
    print(f"Acc: {acc}, Report_acc: {acc_report}, F1: {f1}, Prec: {precision}, Rec: {recall}")
    print(detailed_metrics)
    print("Done")
