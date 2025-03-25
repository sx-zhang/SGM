# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block, DropPath, Mlp

from util.pos_embed import get_2d_sincos_pos_embed
from einops import rearrange
import os

from util.cross_attention import Cross_model

import copy

d_gibson = ["chair", "couch", "potted plant", "bed", "toilet", "tv", "dining-table", "oven", "sink", "refrigerator", "book", "clock", "vase", "cup", "bottle"]

def cal_cls_IOU(input, output):
    n,c,h,w=input.size()
    iou_tmp = torch.zeros(c-2,1,device="cuda")
    iou_num_tmp = torch.zeros(c-2,1,device="cuda")
    for i in range(n):
        for j in range(2,c):
            tmp = input[i,j,:,:] + output[i,j,:,:]
            tmp = torch.where(tmp>0,1.0,0.0)
            if tmp.sum()==0:
                continue
            else:
                iou_tmp[j-2] += (input[i,j,:,:]*output[i,j,:,:]).sum()/tmp.sum()
                iou_num_tmp[j-2] += 1
    return iou_tmp.sum() / iou_num_tmp.sum()

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, 
                 asymmetric_decoder=False, mask_ratio=0.75, vis_mask_ratio=0., cross_layer_num=0):
        super().__init__()

        self.vis_mask_ratio = vis_mask_ratio
        if vis_mask_ratio > 0:
            self.vis_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.llm_features = {}
        for i in range(15):
            self.llm_features[i]=torch.load("roberta_gibson_glm/{}.pt".format(d_gibson[i]))
         
        self.bcn_logits = nn.BCEWithLogitsLoss()
        
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth-cross_layer_num)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # cross attention (zsx added)
        if cross_layer_num>0:
            dim_head = int(embed_dim/num_heads)
            self.cross_blocks = Cross_model(dim=embed_dim, layer_num=cross_layer_num, dim_head=dim_head, heads=num_heads, ff_mult=mlp_ratio)
        else:
            self.cross_blocks = None 
        
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
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
        torch.nn.init.normal_(self.mask_token, std=.02)
        if hasattr(self, 'vis_mask_token'):
            torch.nn.init.normal_(self.vis_mask_token, std=.02)

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
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        c = imgs.shape[1]
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        #x = rearrange(imgs, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        return x

    def unpatchify(self, x, c=3):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward_encoder(self, x, mask):
        # yxy added
        n,c,h,w=x.size()  # 128 17 224 224
        context_infor = torch.zeros(n,128,768,device=x.device)
        for i in range(n):
            a=torch.zeros(1,0,768,device=x.device)
            for j in range(2, c):
                if x[i,j,:,:].max()>0:
                    a=torch.cat((a,self.llm_features[j-2].to(x.device)),1)
            _, d_num, _ = a.size()
            if d_num<128:
                a=torch.cat((a,torch.zeros(1,128-d_num,768,device=x.device)),1)
            context_infor[i,:,:]=a[0,:128,:]
        
        # embed patches
        # print(x.shape)
        # prrint(self.patch_embed)
        x = self.patch_embed(x) # 128 196 768

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        N, _, D = x.shape
        x = x[~mask].reshape(N, -1, D)

        if self.vis_mask_ratio > 0:
            vis_mask_token = self.vis_mask_token + self.pos_embed[:, 1:, :]
            vis_mask_token = vis_mask_token.expand(N, -1, -1)
            vis_mask_token = vis_mask_token[~mask].reshape(N, -1, D)
            L = x.size(1)
            noise = torch.rand(N, L, device=x.device)
            ids_restore = torch.argsort(noise, dim=1)

            len_keep = int(L * (1 - self.vis_mask_ratio))
            vis_mask = torch.ones([N, L], device=x.device)
            vis_mask[:, :len_keep] = 0
            vis_mask = torch.gather(vis_mask, dim=1, index=ids_restore).unsqueeze(-1)

            x = x * (1. - vis_mask) + vis_mask_token * vis_mask

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)  # 128 50 768
            
        # cross attention (zsx added)
        x = self.cross_blocks(x, context_infor)
            
        x = self.norm(x)

        return x


    def forward_decoder(self, x, mask):
        # embed tokens
        x = self.decoder_embed(x)
        x_vis = x[:, 1:, :]
        N, _, D = x_vis.shape

        # append mask tokens to sequence
        expand_pos_embed = self.decoder_pos_embed[:, 1:, :].expand(N, -1, -1)
        pos_vis = expand_pos_embed[~mask].reshape(N, -1, D)
        pos_mask = expand_pos_embed[mask].reshape(N, -1, D)

        x_ = torch.cat([x_vis + pos_vis, self.mask_token + pos_mask], dim=1)

        # add cls_token + decoder_pos_embed
        x = torch.cat([x[:, :1, :] + self.decoder_pos_embed[:, :1, :], x_], dim=1)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x, pos_mask.shape[1]

    
    def forward_loss(self, imgs, pred, mask, pred_iou):
        """
        imgs: [N, 3, H, W]
        pred: [N, mask, p*p*3] 
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        N, _, D = target.shape
        target = target[mask].reshape(N, -1, D)
        target_c = copy.deepcopy(target)
        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6)**.5 # (N, L, p*p*3)
        # channel normalize (sxz)
        count_in_channel = imgs.sum(-1).sum(-1).mean(0)
        channel_mask = torch.where(count_in_channel>5,0,1).bool()
        m = count_in_channel.max()
        chanel_norm = (count_in_channel+1).max()/(count_in_channel+1)
        chanel_norm = chanel_norm.masked_fill(channel_mask, value=torch.tensor(0.0))
        chanel_norm = chanel_norm/chanel_norm.max()
        n,c,h,w = imgs.shape
        chanel_norm_weight = self.patchify(chanel_norm.view(1,c,1,1).repeat(n,1,h,w))
        chanel_norm_weight = chanel_norm_weight[mask].reshape(N, -1, D)
        
        # iou_loss (yxy added)
        h_1 = imgs.size(2) // 16
        B = torch.tensor(range(0,h_1*h_1)).to(imgs.device).unsqueeze(0)
        B = B.repeat(n,1)
        order = mask.float()*1000+B
        ids = torch.argsort(order, dim=1)
        ids_ = torch.argsort(ids, dim=1)
        y = torch.gather(pred_iou, dim=1, index=ids_.unsqueeze(-1).expand(-1, -1, pred_iou.size(-1)))
        y = torch.where(y>0.7,1.0,0.0)
        y = self.unpatchify(y, c=imgs.shape[1])
        y = y.detach()
        iou_loss = (1 - cal_cls_IOU(imgs, y)) / 2
        
        ### channel normalize
        # loss = (pred - target) ** 2 * chanel_norm_weight*100 + iou_loss 
        loss = (pred - target) ** 2 * chanel_norm_weight*100 + self.bcn_logits(pred, target_c)*5 + iou_loss
        loss = loss.mean()
        return loss

    def no_weight_decay(self):
        return {'pos_embed'}

    def forward(self, imgs, mask):
        latent = self.forward_encoder(imgs, mask) # returned mask may change
        pred, mask_num = self.forward_decoder(latent, mask)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred[:, -mask_num:], mask, pred_iou=pred)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(chnum=3, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, in_chans=chnum, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks


def mae_vit_base_patch16_dec512d2b(chnum=3, **kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, in_chans=chnum, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=2, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), cross_layer_num=4, **kwargs)
    return model
