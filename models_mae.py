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
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed1 = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed2 = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed3 = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed4 = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token1 = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token2 = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token3 = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token4 = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks1 = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_blocks2 = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_blocks3 = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_blocks4 = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm1 = norm_layer(decoder_embed_dim)
        self.decoder_norm2 = norm_layer(decoder_embed_dim)
        self.decoder_norm3 = norm_layer(decoder_embed_dim)
        self.decoder_norm4 = norm_layer(decoder_embed_dim)

        self.decoder_pred1 = nn.Linear(decoder_embed_dim, 512, bias=True) # decoder to patch
        self.decoder_pred2 = nn.Linear(decoder_embed_dim, 512, bias=True)  # decoder to patch
        self.decoder_pred3 = nn.Linear(decoder_embed_dim, 512, bias=True)  # decoder to patch
        self.decoder_pred4 = nn.Linear(decoder_embed_dim, 512, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token1, std=.02)
        torch.nn.init.normal_(self.mask_token2, std=.02)
        torch.nn.init.normal_(self.mask_token3, std=.02)
        torch.nn.init.normal_(self.mask_token4, std=.02)

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
        count = 0
        f=[]
        for blk in self.blocks:
            count+=1
            x = blk(x)
            if count==6:
                f.append(x)
            if count==8:
                f.append(x)
            if count==10:
                f.append(x)
        x = self.norm(x)

        return f[0], f[1], f[2], x, mask, ids_restore

    def forward_decoder1(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed1(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token1.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks1:
            x = blk(x)
        x = self.decoder_norm1(x)

        # predictor projection
        x = self.decoder_pred1(x)
        return x

    def forward_decoder2(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed2(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token2.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks2:
            x = blk(x)
        x = self.decoder_norm2(x)

        # predictor projection
        x = self.decoder_pred2(x)
        return x
    def forward_decoder3(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed3(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token3.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks3:
            x = blk(x)
        x = self.decoder_norm3(x)

        # predictor projection
        x = self.decoder_pred3(x)
        return x
    def forward_decoder4(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed4(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token4.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks4:
            x = blk(x)
        x = self.decoder_norm4(x)

        # predictor projection
        x = self.decoder_pred4(x)
        return x

    def forward_loss(self, pred, teacher_out, mask):
        pred = pred / pred.norm(dim=2, keepdim=True)
        teacher_out = teacher_out / teacher_out.norm(dim=2, keepdim=True)
        assert pred.shape == teacher_out.shape
        loss = 2 - 2 * (pred * teacher_out).sum(dim=-1)
        loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def forward(self, imgs, teacher_out, mask_ratio=0.75):
        latent6, latent8, latent10, latent12, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred1 = self.forward_decoder1(latent6, ids_restore)
        pred2 = self.forward_decoder2(latent8, ids_restore)
        pred3 = self.forward_decoder3(latent10, ids_restore)
        pred4 = self.forward_decoder4(latent12, ids_restore)
        loss1 = self.forward_loss(pred1[:, 1:], teacher_out[:, 1:], mask)
        loss2 = self.forward_loss(pred2[:, 1:], teacher_out[:, 1:], mask)
        loss3 = self.forward_loss(pred3[:, 1:], teacher_out[:, 1:], mask)
        loss4 = self.forward_loss(pred4[:, 1:], teacher_out[:, 1:], mask)
        return loss1, loss2, loss3, loss4


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=4, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
