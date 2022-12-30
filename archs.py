import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from utils import *

__all__ = ['UNext']

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import types
import math
from abc import ABCMeta, abstractmethod
from mmcv.cnn import ConvModule
import pdb
import settings


class _MatrixDecomposition2DBase(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.spatial = getattr(args, 'SPATIAL', True)

        self.S = getattr(args, 'MD_S', 1)
        self.D = getattr(args, 'MD_D', 512)
        self.R = getattr(args, 'MD_R', 64)

        self.train_steps = getattr(args, 'TRAIN_STEPS', 2)
        self.eval_steps = getattr(args, 'EVAL_STEPS', 2)

        self.inv_t = getattr(args, 'INV_T', 100)
        self.eta = getattr(args, 'ETA', 0.9)

        self.rand_init = getattr(args, 'RAND_INIT', True)

        print('spatial', self.spatial)
        print('S', self.S)
        print('D', self.D)
        print('R', self.R)
        print('train_steps', self.train_steps)
        print('eval_steps', self.eval_steps)
        print('inv_t', self.inv_t)
        print('eta', self.eta)
        print('rand_init', self.rand_init)

    def _build_bases(self, B, S, D, R, cuda=False):
        raise NotImplementedError

    def local_step(self, x, bases, coef):
        raise NotImplementedError

    @torch.no_grad()
    def local_inference(self, x, bases):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.inv_t * coef, dim=-1)

        steps = self.train_steps if self.training else self.eval_steps
        for _ in range(steps):  # 多次矩阵分解，多次求Base、coef矩阵
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        raise NotImplementedError

    def forward(self, x, return_bases=False):
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        if not self.rand_init and not hasattr(self, 'bases'):
            bases = self._build_bases(1, self.S, D, self.R, cuda=True)
            self.register_buffer('bases', bases)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self._build_bases(B, self.S, D, self.R, cuda=True)
        else:
            bases = self.bases.repeat(B, 1, 1)
        # 以下调用矩阵分解方法，求出系数矩阵和基矩阵
        bases, coef = self.local_inference(x, bases)

        # (B * S, N, R)
        coef = self.compute_coef(x, bases, coef)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        # 返回最终的分解好的大矩阵
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        # 恢复矩阵在网络中的形状
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)

        # (B * H, D, R) -> (B, H, N, D)
        bases = bases.view(B, self.S, D, self.R)

        if not self.rand_init and not self.training and not return_bases:
            self.online_update(bases)

        # if not self.rand_init or return_bases:
        #     return x, bases
        # else:
        return x

    @torch.no_grad()
    def online_update(self, bases):
        # (B, S, D, R) -> (S, D, R)
        update = bases.mean(dim=0)
        self.bases += self.eta * (update - self.bases)
        self.bases = F.normalize(self.bases, dim=1)


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(self, args):
        super().__init__(args)

        self.inv_t = 1

    def _build_bases(self, B, S, D, R, cuda=False):
        if cuda:
            bases = torch.rand((B * S, D, R)).cuda()
        else:
            bases = torch.rand((B * S, D, R))

        bases = F.normalize(bases, dim=1)

        return bases

    @torch.no_grad()
    def local_step(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)  # 分子
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))  # 分母
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)  # 系数矩阵

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coef)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)  # 基矩阵

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)

        return coef


def get_hams(key):
    # hams = {'VQ':VQ2D,
    #         'CD':CD2D,
    #         'NMF':NMF2D}
    hams = {
        'NMF': NMF2D}
    assert key in hams

    return hams[key]


##################################################################################


class EfficientSelfAttenHam(nn.Module):
    def __init__(self, dim, args):
        super().__init__()
        HAM = get_hams('NMF')
        self.ham = HAM(args)  # 调用NMF2D，并把args传递给HAM
        self.dw = DWConv(dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # (B,C,H,W)
        _, _, H, W, = x.shape
        x = self.ham(x)
        x = x.reshape(B, N, -1).contiguous()  # (B,C,N)

        x = self.dw(x, H, W)  # 加上位置编码后再继续
        out = self.proj(x)

        return out


##################################################################################

class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W) + self.fc1(x)))
        out = self.fc2(ax)
        return out


class TransformerBlockHam(nn.Module):
    def __init__(self, dim, token_mlp='mix'):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.act = nn.ReLU(inplace=True)
        self.attn = EfficientSelfAttenHam(dim, settings)
        self.norm2 = nn.LayerNorm(dim)
        if token_mlp == 'mix':
            self.mlp = MixFFN(dim, int(dim * 4))
        elif token_mlp == 'mix_skip':
            self.mlp = MixFFN_skip(dim, int(dim * 4))
        else:
            self.mlp = MLP_FFN(dim, int(dim * 4))

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:

        tx = x + self.attn(self.act(self.norm1(x)), H, W)  # 先LayerNorm，再relu后，再送入矩阵分解模块
        mx = tx + self.mlp(self.norm2(tx), H, W)
        return mx


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


def shift(dim):
    x_shift = [torch.roll(x_c, shift, dim) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
    x_cat = torch.cat(x_shift, 1)
    x_cat = torch.narrow(x_cat, 2, self.pad, H)
    x_cat = torch.narrow(x_cat, 3, self.pad, W)
    return x_cat


class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        # 把xn在第1个维度(通道维度)上切成shift_size = 5个块，返回值是一个元祖，里面存的是所有的tensor块
        xs = torch.chunk(xn, self.shift_size, 1)
        # 挤牙膏、移动特征
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        # 在第一个维度上，拼接特征
        x_cat = torch.cat(x_shift, 1)

        '''
        torch.narrow(input, dim, start, length)函数用法：
        input– 待处理的tensor
        dim – 维度，当为0时以行为单位进行切片，当为1时以列为单位进行切片
        start – 切片开始的索引
        length – 切片的长度
        '''
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)

        ##到此处，特征处理完毕，以下恢复特征形状，并且加入位置编码以及各种优化层
        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_r = x_s.transpose(1, 2)
        x = self.fc1(x_shift_r)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)

        # y轴的MLP，在上一步的后面继续做一次滑动
        xn = x.transpose(1, 2).view(B, C, H, W).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)

        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)

        x_s = x_s.reshape(B, C, H * W).contiguous()
        x_shift_c = x_s.transpose(1, 2)
        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x


class shiftedBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # 第一次调用：输入图片尺寸：64，经过的卷积：输入通道128，输出通道160，卷积核大小为3，步长2,pad是1；
        # 输出图片尺寸：C是160，H和W是32，B不变
        x = self.proj(x)
        _, _, H, W = x.shape

        # print("##########################")
        # print(x.shape)

        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


################################自注意力消融实验#########################################

class M_EfficientSelfAtten(nn.Module):
    def __init__(self, dim, head, reduction_ratio):
        super().__init__()
        self.head = head
        self.reduction_ratio = reduction_ratio  # list[1  2  4  8]
        self.scale = (dim // head) ** -0.5
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim)

        if reduction_ratio is not None:
            self.scale_reduce = Scale_reduce(dim, reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.head, C // self.head).permute(0, 2, 1, 3)

        if self.reduction_ratio is not None:
            x = self.scale_reduce(x)

        kv = self.kv(x).reshape(B, -1, 2, self.head, C // self.head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn_score = attn.softmax(dim=-1)

        x_atten = (attn_score @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(x_atten)

        return out


class Scale_reduce(nn.Module):
    def __init__(self, dim, reduction_ratio):
        super().__init__()
        self.dim = dim
        self.reduction_ratio = reduction_ratio
        if (len(self.reduction_ratio) == 4):
            self.sr0 = nn.Conv2d(dim, dim, reduction_ratio[3], reduction_ratio[3])
            self.sr1 = nn.Conv2d(dim * 2, dim * 2, reduction_ratio[2], reduction_ratio[2])
            self.sr2 = nn.Conv2d(dim * 5, dim * 5, reduction_ratio[1], reduction_ratio[1])

        elif (len(self.reduction_ratio) == 3):
            self.sr0 = nn.Conv2d(dim * 2, dim * 2, reduction_ratio[2], reduction_ratio[2])
            self.sr1 = nn.Conv2d(dim * 5, dim * 5, reduction_ratio[1], reduction_ratio[1])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        if (len(self.reduction_ratio) == 4):
            tem0 = x[:, :3136, :].reshape(B, 56, 56, C).permute(0, 3, 1, 2)
            tem1 = x[:, 3136:4704, :].reshape(B, 28, 28, C * 2).permute(0, 3, 1, 2)
            tem2 = x[:, 4704:5684, :].reshape(B, 14, 14, C * 5).permute(0, 3, 1, 2)
            tem3 = x[:, 5684:6076, :]

            sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
            sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)
            sr_2 = self.sr2(tem2).reshape(B, C, -1).permute(0, 2, 1)

            reduce_out = self.norm(torch.cat([sr_0, sr_1, sr_2, tem3], -2))

        if (len(self.reduction_ratio) == 3):
            tem0 = x[:, :1568, :].reshape(B, 28, 28, C * 2).permute(0, 3, 1, 2)
            tem1 = x[:, 1568:2548, :].reshape(B, 14, 14, C * 5).permute(0, 3, 1, 2)
            tem2 = x[:, 2548:2940, :]

            sr_0 = self.sr0(tem0).reshape(B, C, -1).permute(0, 2, 1)
            sr_1 = self.sr1(tem1).reshape(B, C, -1).permute(0, 2, 1)

            reduce_out = self.norm(torch.cat([sr_0, sr_1, tem2], -2))

        return reduce_out


class UNext(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self, num_classes, input_channels=3, deep_supervision=False, img_size=224, patch_size=16, in_chans=3,
                 embed_dims=[128, 160, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1], **kwargs):
        super().__init__()

        self.encoder1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.encoder2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.encoder3 = nn.Conv2d(32, 128, 3, stride=1, padding=1)

        self.ebn1 = nn.BatchNorm2d(16)
        self.ebn2 = nn.BatchNorm2d(32)
        self.ebn3 = nn.BatchNorm2d(128)

        self.norm3 = norm_layer(embed_dims[1])
        self.norm4 = norm_layer(embed_dims[2])

        self.dnorm3 = norm_layer(160)
        self.dnorm4 = norm_layer(128)

        # self.addnorm1 = nn.LayerNorm(128)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # self.block1 = nn.ModuleList([shiftedBlock(
        #     dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
        #     sr_ratio=sr_ratios[0])])
        self.block1 = nn.ModuleList([
            # TransformerBlock(dims[0], heads[0], reduction_ratios[0],token_mlp)
            TransformerBlockHam(embed_dims[1], 'mix_skip')  # 通道数必须改成64
        ])

        # self.block2 = nn.ModuleList([shiftedBlock(
        #     dim=embed_dims[2], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
        #     sr_ratio=sr_ratios[0])])
        self.block2 = nn.ModuleList([
            # TransformerBlock(dims[0], heads[0], reduction_ratios[0],token_mlp)
            TransformerBlockHam(embed_dims[2], 'mix_skip')  # 通道数必须改成64
        ])

        # self.dblock1 = nn.ModuleList([shiftedBlock(
        #     dim=embed_dims[1], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[0], norm_layer=norm_layer,
        #     sr_ratio=sr_ratios[0])])
        self.dblock1 = nn.ModuleList([
            # TransformerBlock(dims[0], heads[0], reduction_ratios[0],token_mlp)
            TransformerBlockHam(embed_dims[1], 'mix_skip')  # 通道数必须改成64
        ])

        # self.dblock2 = nn.ModuleList([shiftedBlock(
        #     dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=1, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[1], norm_layer=norm_layer,
        #     sr_ratio=sr_ratios[0])])
        self.dblock2 = nn.ModuleList([
            # TransformerBlock(dims[0], heads[0], reduction_ratios[0],token_mlp)
            TransformerBlockHam(embed_dims[0], 'mix_skip')  # 通道数必须改成64
        ])

        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv2d(256, 160, 3, stride=1, padding=1)
        self.decoder2 = nn.Conv2d(160, 128, 3, stride=1, padding=1)
        self.decoder3 = nn.Conv2d(128, 32, 3, stride=1, padding=1)
        self.decoder4 = nn.Conv2d(32, 16, 3, stride=1, padding=1)
        self.decoder5 = nn.Conv2d(16, 16, 3, stride=1, padding=1)

        self.dbn1 = nn.BatchNorm2d(160)
        self.dbn2 = nn.BatchNorm2d(128)
        self.dbn3 = nn.BatchNorm2d(32)
        self.dbn4 = nn.BatchNorm2d(16)

        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

        self.soft = nn.Softmax(dim=1)

        self.changeC1 = conv1x1(16, 128)
        self.changeC2 = conv1x1(32, 128)
        self.changeC3 = conv1x1(128, 128)
        self.changeC4 = conv1x1(160, 128)
        self.changeC5 = conv1x1(256, 128)

        self.restore1 = conv1x1(128, 16)
        self.restore2 = conv1x1(128, 32)
        self.restore3 = conv1x1(128, 128)
        self.restore4 = conv1x1(128, 160)
        self.restore5 = conv1x1(128, 256)
        self.normrestore = nn.LayerNorm(128)

        self.mixfeature = TransformerBlockHam(128, 'mix_skip')
        self.reduction_ratios = None
        self.addfeature = M_EfficientSelfAtten(160, 1, self.reduction_ratios)
        self.addfeature2 = M_EfficientSelfAtten(256, 1, self.reduction_ratios)
        self.addfeature3 = M_EfficientSelfAtten(128, 1, self.reduction_ratios)

    def forward(self, x):

        B = x.shape[0]
        ### Encoder
        ### Conv Stage
        Container = []

        # 卷积输出图片尺寸：(224-3+2)/1  + 1 = 224,池化输出尺寸：112，也是输出尺寸
        out = F.relu(F.max_pool2d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        Container.append(t1)  # 112

        # 通道数变成：32；图片尺寸变成：56；
        out = F.relu(F.max_pool2d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        Container.append(t2)  # 56

        ### Stage 3：
        # 通道数变成：128；图片尺寸变成：28；
        out = F.relu(F.max_pool2d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out
        Container.append(t3)  # 28

        # #################################分割线：上面提取局部特征，下面提取全局特征###########################
        # 图片尺寸变成14，通道变成160，先改变图片尺寸，再送入矩阵分解模块
        out, H, W = self.patch_embed3(out)
        # 矩阵分解
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W)
        out = self.norm3(out)
        addout = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t4 = addout
        Container.append(t4)  # 14
        # 自注意力
        addout = self.addfeature(out)
        addout = self.norm3(addout)
        # addoutshape1 = addout.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = addout
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        tmpt4 = out
        Container.append(tmpt4)

        ### Bottleneck
        # 图片尺寸变成7，通道数变成256
        out, H, W = self.patch_embed4(out)
        # 矩阵分解
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = self.norm4(out)
        addoutshape = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        t5 = addoutshape
        Container.append(t5)  # 矩阵分解结果加入容器
        # 自注意力
        addout2 = self.addfeature2(out)
        addout2 = self.norm4(addout2)
        # out = torch.add(addout2, out)
        out = addout2
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        tmpt5 = out
        Container.append(tmpt5)
        # print(Container[0].shape)   #torch.Size([16, 16, 112, 112])
        # print(Container[1].shape)   #torch.Size([16, 32, 56, 56])
        # print(Container[2].shape)   #torch.Size([16, 128, 28, 28])
        # print(Container[3].shape)   #torch.Size([16, 160, 14, 14])
        # print(Container[4].shape)   #torch.Size([16, 256, 7, 7])  (B、C、H、W)

        conOut1 = self.changeC1(Container[0])  # t1   112
        conOut2 = self.changeC2(Container[1])  # t2   56
        conOut3 = self.changeC3(Container[2])  # t3   28
        conOut4 = self.changeC4(Container[3])  # t4     14
        conOut5 = self.changeC4(Container[4])  # tmpt4  14
        conOut6 = self.changeC5(Container[5])  # t5     7
        conOut7 = self.changeC5(Container[6])  # tmpt5  7
        # print(conOut1.shape)   #torch.Size([16, 128, 112, 112])
        # print(conOut2.shape)   #torch.Size([16, 128, 56, 56])
        # print(conOut3.shape)   #torch.Size([16, 128, 28, 28])
        # print(conOut4.shape)   #torch.Size([16, 128, 14, 14])
        # print(conOut5.shape)   #torch.Size([16, 128, 7, 7])

        c1f = conOut1.permute(0, 2, 3, 1).reshape(B, -1, 128)
        c1f = self.normrestore(c1f)
        c2f = conOut2.permute(0, 2, 3, 1).reshape(B, -1, 128)
        c2f = self.normrestore(c2f)
        c3f = conOut3.permute(0, 2, 3, 1).reshape(B, -1, 128)
        c3f = self.normrestore(c3f)
        c4f = conOut4.permute(0, 2, 3, 1).reshape(B, -1, 128)
        c4f = self.normrestore(c4f)
        c5f = conOut5.permute(0, 2, 3, 1).reshape(B, -1, 128)
        c5f = self.normrestore(c5f)
        c6f = conOut6.permute(0, 2, 3, 1).reshape(B, -1, 128)
        c6f = self.normrestore(c6f)
        c7f = conOut7.permute(0, 2, 3, 1).reshape(B, -1, 128)
        c7f = self.normrestore(c7f)
        # print(c1f.shape)   #torch.Size([16, 12544, 128])
        # print(c2f.shape)   #torch.Size([16, 3136, 128])
        # print(c3f.shape)   #torch.Size([16, 784, 128])
        # print(c4f.shape)   #torch.Size([16, 196, 128])
        # print(c5f.shape)   #torch.Size([16, 196, 128])
        # print(c6f.shape)   #torch.Size([16, 49, 128])
        # print(c7f.shape)   #torch.Size([16, 49, 128])

        seq = torch.cat([c1f, c2f, c3f, c4f, c5f, c6f, c7f], -2)
        # print(seq.shape)   #torch.Size([16, 16954, 128])
        seqout = self.mixfeature(seq, 16954, 1)
        seqout = self.normrestore(seqout)
        # print(seqout.shape)   #torch.Size([16, 16709, 128])

        seqfinal = seq + seqout
        seqfinal = self.normrestore(seqfinal)

        tem1 = seqfinal[:, :12544, :].reshape(B, -1, 128)  # 112
        tem1 = self.normrestore(tem1)
        tem2 = seqfinal[:, 12544:15680, :].reshape(B, -1, 128)  # 56
        tem2 = self.normrestore(tem2)
        tem3 = seqfinal[:, 15680:16464, :].reshape(B, -1, 128)  # 28
        tem3 = self.normrestore(tem3)
        tem4 = seqfinal[:, 16464:16660, :].reshape(B, -1, 128)  # 14
        tem4 = self.normrestore(tem4)
        tem5 = seqfinal[:, 16660:16856, :].reshape(B, -1, 128)  # 14
        tem5 = self.normrestore(tem5)
        tem6 = seqfinal[:, 16856:16905, :].reshape(B, -1, 128)  # 7
        tem6 = self.normrestore(tem6)
        tem7 = seqfinal[:, 16905:, :].reshape(B, -1, 128)
        tem7 = self.normrestore(tem7)

        t1 = tem1.reshape(B, 112, 112, 128).permute(0, 3, 1, 2).contiguous()
        t2 = tem2.reshape(B, 56, 56, 128).permute(0, 3, 1, 2).contiguous()
        t3 = tem3.reshape(B, 28, 28, 128).permute(0, 3, 1, 2).contiguous()
        t4 = tem4.reshape(B, 14, 14, 128).permute(0, 3, 1, 2).contiguous()
        t5 = tem5.reshape(B, 14, 14, 128).permute(0, 3, 1, 2).contiguous()
        t6 = tem6.reshape(B, 7, 7, 128).permute(0, 3, 1, 2).contiguous()
        t7 = tem7.reshape(B, 7, 7, 128).permute(0, 3, 1, 2).contiguous()
        t1 = self.restore1(t1)
        t2 = self.restore2(t2)
        t3 = self.restore3(t3)
        t4 = self.restore4(t4)
        t5 = self.restore4(t5)
        t6 = self.restore5(t6)
        t7 = self.restore5(t7)
        # print(t1.shape)   #torch.Size([16, 112, 112, 128])
        # print(t2.shape)   #torch.Size([16, 56, 56, 128])
        # print(t3.shape)   #torch.Size([16, 28, 28, 128])
        # print(t4.shape)   #torch.Size([16, 14, 14, 128])
        # print(t5.shape)   #torch.Size([16, 7, 7, 128])

        ################################编码器结束，以下是解码器########################################

        t7 = torch.add(t7, t6)  # 自注意力->矩阵分解
        ### Stage 4
        # 瓶颈层先上采样，得到和上一层特征图尺寸一样的图14 160
        out = F.relu(F.interpolate(self.dbn1(self.decoder1(t7)), scale_factor=(2, 2), mode='bilinear'))
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        # 自注意力
        addout = self.addfeature(out)
        addout = self.dnorm3(addout)
        addout = addout.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = torch.add(addout, t5)
        # 矩阵分解
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        out = torch.add(out, t4)

        # 然后如上述一样依次进行  28
        out = F.relu(F.interpolate(self.dbn2(self.decoder2(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t3)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W)

        out = self.dnorm4(out)
        addout = self.addfeature3(out)
        addout = self.normrestore(addout)
        out = addout
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # 56
        out = F.relu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t2)
        # 112
        out = F.relu(F.interpolate(self.dbn4(self.decoder4(out)), scale_factor=(2, 2), mode='bilinear'))
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2), mode='bilinear'))

        return self.final(out)


if __name__ == '__main__':
    x = torch.randn(2, 3, 224, 224).cuda(0)
    net = UNext(3, 3, False).cuda(0)

    print(net(x).shape)
    print("Total number of parameters in networks is {} M".format(sum(x.numel() for x in net.parameters()) / 1e6))