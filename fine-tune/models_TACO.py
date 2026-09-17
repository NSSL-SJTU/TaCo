# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

# Modified for TaCo:
# Learning Flow Semantics for Encrypted Traffic Analysis:
# A Contrastive Pre-training Approach.

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.models.vision_transformer import Block, DropPath, Mlp



class TaCoEncoder_finetune(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, global_pool=False, **kwargs):
        super(TaCoEncoder_finetune, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def packet_window_attention(self, x, cls_token, block_index):
        blk = self.blocks[block_index]
        N, L, D = x.shape  # batch, length, dim
        packet_count = 5 # partition window count (packet count)
        cls_tokens = cls_token.expand(N * packet_count, -1, -1)
        packets_x = x.reshape(N * packet_count, -1, D)
        packets_x = torch.cat((cls_tokens, packets_x), dim=1)
        packets_x = blk(packets_x)
        x = packets_x[:, 1:, :].reshape(N, -1, D)

        return x, cls_tokens

    def flow_window_attention(self, x, cls_token, block_index):
        blk = self.blocks[block_index]
        N, L, D = x.shape  # batch, length, dim
        count = 4 # partition window count
        cls_tokens = cls_token.expand(N * count, -1, -1)
        if self.training:
            packets_x = x.reshape(N, 5, -1, D)
            noise = torch.rand(N, packets_x.shape[2], device=x.device)  # noise in [0, 1] 
            # sort noise for each sample
            packet_ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove 
        else:
            # x_grouped = x.reshape(N, count, -1, D).transpose(1, 2).reshape(N*count, -1, D)  # shuffle
            packets_x = x.reshape(N, 5, -1, D)
            packet_ids = torch.arange(0, packets_x.shape[2], device=x.device).repeat(N, 1)  # N, packet tokens number
            packet_ids_shuffle = packet_ids.reshape(N, count, -1).transpose(1, 2).flatten(1)

        miss_indices = torch.arange(0, L, packets_x.shape[2], device=x.device).repeat(N, packets_x.shape[2],
                                                                                      1).transpose(1,
                                                                                                   2)  # N, packet_count,L
        ids_shuffle = packet_ids_shuffle.repeat(1, packets_x.shape[1]) + miss_indices.reshape(N, -1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        x_grouped = torch.gather(x, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, D)).reshape(N * count, -1,
                                                                                                    D)

        # print(x_grouped.shape)
        x_grouped = torch.cat((cls_tokens, x_grouped), dim=1)

        x_grouped = blk(x_grouped)

        x = x_grouped[:, 1:, :].reshape(N, -1, D)

        x = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        return x, cls_tokens


    def forward_features(self, x):
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        N, L, D = x.shape  # batch, length, dim
        count = 4

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        # x = torch.cat((cls_tokens, x), dim=1)

        x, cls_tokens = self.packet_window_attention(x, cls_token, 0)
        x, cls_tokens = self.flow_window_attention(x, cls_token, 1)
        x, cls_tokens = self.packet_window_attention(x, cls_token, 2)
        x, cls_tokens = self.flow_window_attention(x, cls_token, 3)

        cls_tokens = cls_tokens.reshape(N, count, -1, D).mean(dim=1)
        x = torch.cat((cls_tokens, x), dim=1)
        #print(x.shape)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome



def TaCo_classifier(**kwargs):
    model = TaCoEncoder_finetune(
        img_size=40, patch_size=2, in_chans=1, embed_dim=192, depth=4, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


