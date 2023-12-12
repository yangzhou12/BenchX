# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial


import torch
import torchvision
import torch.nn as nn
from torchvision.transforms.functional import InterpolationMode
from timm.models.vision_transformer import PatchEmbed, Block

from .util.pos_embed import get_2d_sincos_pos_embed
from .bert.bert_encoder import BertEncoder

class MRM(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        #Look at ViTMAE implementation for image encoder-decoder workings

        # --------------------------------------------------------------------------
        # image encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim) #image to patch embedding 
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)]) #Transformer block layer
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # image decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, (patch_size*2)**2 * in_chans, bias=True)
        # --------------------------------------------------------------------------
        # Bert encoder
        self.bert_encoder = BertEncoder()
        self.bert_mlp = nn.Linear(embed_dim, 384, bias=True)
        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        ## Note: Get position embeddings for each patch
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0)) #np.data -> Python buffer object pointing to the start of the array's data

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data #store weights array
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1])) #weights are initialized via the normalized Xavier weight initialization

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """

        p = self.patch_embed.patch_size[0]*2
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0] * 2
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore, projection=True):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        if projection:
            x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_report_decoder(self, latent, caption_ids, labels, attention_mask, token_type_ids):
        latent = self.bert_mlp(latent)
        latent = latent[:, 1:, :].mean(dim=1)
        outputs = self.bert_encoder(latent, caption_ids, labels, attention_mask, token_type_ids)
        return outputs

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, batch, mask_ratio=0.75):
        big_imgs = batch["image"]
        
        ids, labels, attention_mask, type_ids = batch["ids"], batch["labels"], batch["attention_mask"], batch["type_ids"]

        big_imgs = big_imgs.cuda()
        ids = ids.cuda()
        labels = labels.cuda()
        attention_mask = attention_mask.cuda()
        type_ids = type_ids.cuda()
        imgs = torchvision.transforms.Resize([224,224], interpolation=InterpolationMode.BICUBIC)(big_imgs)

        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        outputs = self.forward_report_decoder(latent, ids, labels, attention_mask, type_ids)
        report_loss = outputs["loss"]
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(big_imgs, pred, mask)
        return (loss, report_loss), pred, mask
    
    def forward_embeddings(self, imgs=None, texts=None):
        imgs = imgs.cuda()
        input_ids = texts["input_ids"].cuda()
        attention_mask = texts["attention_mask"].cuda()
        token_type_ids = texts["token_type_ids"].cuda()

        img_emb_g = []
        text_emb_g = []

        # 1 Image 5 prompts for 5 classes
        # Cos sim img in [1,384] text [5,5,384] - [1000, 384]; [25, 384]
        # [1, 5] for the prompts of certain class, avg to get the class score.
        # 1000 images, [1000,5] for 5 classes

        for img in imgs:
            N = input_ids.shape[0]
            latent, _, ids_restore = self.forward_encoder(img.unsqueeze(0), mask_ratio=0) # set mask ratio to 0
            latent, ids_restore = latent.repeat(N, 1, 1), ids_restore.repeat(N, 1)
            outputs = self.forward_report_decoder(latent, input_ids, None, attention_mask, token_type_ids) # set labels to None
            
            pred = self.forward_decoder(latent, ids_restore, projection=False)
            img_embeds = self.bert_mlp(pred)[:, 1:, :].mean(dim=1) # [1, 384]
            text_embeds = outputs["pooler_output"] # [5, 384]

            img_emb_g.append(img_embeds)
            text_emb_g.append(text_embeds)
            
        img_emb_g = torch.mean(torch.stack(img_emb_g), dim=1)
        text_emb_g = torch.mean(torch.stack(text_emb_g), dim=0) # [1000, 5, 384]
        
        return {"img_emb_g": img_emb_g, "text_emb_g": text_emb_g} #[1000, 384] [5, 384]


def mrm(**kwargs):
    model = MRM(
        patch_size=16, in_chans=3, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=768, decoder_depth=4, decoder_num_heads=6,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model