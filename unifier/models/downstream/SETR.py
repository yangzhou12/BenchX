""" Adapted from SETR Implementation

    Code can be found here:
    https://github.com/gupta-abhay/setr-pytorch
"""

import torch
import torch.nn as nn
from unifier.blocks.vision import *


class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=5000):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        return x + position_embeddings


class SegmentationTransformer(nn.Module):
    def __init__(
        self,
        cnn,
        img_dim,
        patch_dim,
        num_channels,
        embedding_dim,  # transformer setting
        num_heads,  # transformer setting
        num_layers,  # transformer setting
        hidden_dim,  # transformer setting
        dropout_rate=0.0,  # transformer setting
        attn_dropout_rate=0.0,  # transformer setting
        conv_patch_representation=False,
        positional_encoding_type="learned",
    ):
        super(SegmentationTransformer, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation

        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels

        self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        ######## Initialize ViT Encoder here ########

        # self.transformer = TransformerModel(
        #     embedding_dim,
        #     num_layers,
        #     num_heads,
        #     hidden_dim,
        #     self.dropout_rate,
        #     self.attn_dropout_rate,
        # )

        cnn_func = cnn.pop("proto")
        self.transformer = eval(cnn_func)(**cnn)

        ####### END #######

        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        if self.conv_patch_representation:
            self.conv_x = nn.Conv2d(
                self.num_channels,
                self.embedding_dim,
                kernel_size=(self.patch_dim, self.patch_dim),
                stride=(self.patch_dim, self.patch_dim),
                padding=self._get_padding(
                    "VALID",
                    (self.patch_dim, self.patch_dim),
                ),
            )
        else:
            self.conv_x = None

    def _init_decode(self):
        raise NotImplementedError("Should be implemented in child class!")

    def encode(self, x):
        n, c, h, w = x.shape
        if self.conv_patch_representation:
            # combine embedding w/ conv patch distribution
            x = self.conv_x(x)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.embedding_dim)
        else:
            x = (
                x.unfold(2, self.patch_dim, self.patch_dim)
                .unfold(3, self.patch_dim, self.patch_dim)
                .contiguous()
            )
            x = x.view(n, c, -1, self.patch_dim**2)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.flatten_dim)
            x = self.linear_encoding(x)

        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # apply transformer
        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)

        return x, intmd_x

    def decode(self, x):
        raise NotImplementedError("Should be implemented in child class!!")

    def forward(self, x, auxillary_output_layers=None):
        encoder_output, intmd_encoder_outputs = self.encode(x)
        decoder_output = self.decode(
            encoder_output, intmd_encoder_outputs, auxillary_output_layers
        )

        if auxillary_output_layers is not None:
            auxillary_outputs = {}
            for i in auxillary_output_layers:
                val = str(2 * i - 1)
                _key = "Z" + str(i)
                auxillary_outputs[_key] = intmd_encoder_outputs[val]

            return decoder_output, auxillary_outputs

        return decoder_output

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ["SAME", "VALID"]
        if padding_type == "SAME":
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class SETR_Naive(SegmentationTransformer):
    def __init__(
        self,
        cnn,
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    ):
        dropout_rate = cnn.pop("hidden_dropout_prob", 0.0)

        super(SETR_Naive, self).__init__(
            cnn=cnn,
            img_dim=img_dim,
            patch_dim=patch_dim,
            num_channels=num_channels,
            dropout_rate=dropout_rate,
            conv_patch_representation=conv_patch_representation,
            positional_encoding_type=positional_encoding_type,
        )

        self.num_classes = num_classes
        self._init_decode()

    def _init_decode(self):
        self.conv1 = nn.Conv2d(
            in_channels=self.embedding_dim,
            out_channels=self.embedding_dim,
            kernel_size=1,
            stride=1,
            padding=self._get_padding(
                "VALID",
                (1, 1),
            ),
        )
        self.bn1 = nn.BatchNorm2d(self.embedding_dim)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=self.embedding_dim,
            out_channels=self.num_classes,
            kernel_size=1,
            stride=1,
            padding=self._get_padding(
                "VALID",
                (1, 1),
            ),
        )
        self.upsample = nn.Upsample(scale_factor=self.patch_dim, mode="bilinear")

    def decode(self, x, intmd_x, intmd_layers=None):
        x = self._reshape_output(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x
