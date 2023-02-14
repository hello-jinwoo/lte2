# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

# from bicubic_pytorch import core

from models import register

'''
edsr_sa -> v2 : learnable slot
'''

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


url = {
    'r16f64x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x2-1bc95232.pt',
    'r16f64x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x3-abf2a44e.pt',
    'r16f64x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_baseline_x4-6b446fab.pt',
    'r32f256x2': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x2-0edfb8a3.pt',
    'r32f256x3': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x3-ea3ef2c6.pt',
    'r32f256x4': 'https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt'
}


class SlotAttention(nn.Module):
    def __init__(self, args, eps=1e-8):
        super().__init__()
        """Builds the Slot Attention module
        Args:
        num_slots (K): Number of slots in Slot Attention.
        iterations: Number of iterations in Slot Attention.
        """
        self.args = args
        self.num_slots = args.slot_num
        self.iterations = args.slot_iters
        self.num_heads = args.slot_attn_heads
        self.input_dim = args.slot_input_dim
        self.slot_dim = args.slot_dim
        self.mlp_hid_dim = args.slot_mlp_hid_dim
        self.scale = (args.slot_dim // args.slot_attn_heads) ** -0.5
        self.eps = eps

        # slot initialize
        if args.slot_init_mode == 'gaussian':
            self.slots_mu = nn.Parameter(torch.randn(1, 1, self.slot_dim))
            self.slots_sigma = nn.Parameter(torch.abs(torch.randn(1, 1, self.slot_dim)))
        elif args.slot_init_mode == 'learnable':
            self.slots = nn.Parameter(torch.randn(1, 1, self.slot_dim))

        self.norm_input = nn.LayerNorm(self.input_dim)
        self.norm_slots = nn.LayerNorm(self.slot_dim)
        self.norm_mlp = nn.LayerNorm(self.slot_dim)

        self.to_q = nn.Linear(self.slot_dim, self.slot_dim)
        self.to_k = nn.Linear(self.input_dim, self.slot_dim)
        self.to_v = nn.Linear(self.input_dim, self.slot_dim)
        self.to_illum = nn.Linear(self.slot_dim,2)

        self.gru = nn.GRUCell(self.slot_dim, self.slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(self.slot_dim, self.mlp_hid_dim), nn.ReLU(), nn.Linear(self.mlp_hid_dim, self.slot_dim)
        )

    def forward(self, inputs, num_slots=None):
        """
        Args
        inputs : [B, N (W x H), D]
        outputs: 
        """
        B, N_in, D_in = inputs.shape
        K = num_slots if num_slots is not None else self.num_slots
        D_slot = self.slot_dim
        N_heads = self.num_heads

        # original initialize of slots
        if self.args.slot_init_mode == 'gaussian':
            mu = self.slots_mu.expand(B, K, -1)         # [B,K,slot_dim]
            sigma = self.slots_sigma.expand(B, K, -1)   # [B,K,slot_dim]
            slots = torch.normal(mu, sigma)             # [B,K,slot_dim]
        elif self.args.slot_init_mode == 'learnable':
            slots = self.slots.expand(B, K, -1)

        inputs = self.norm_input(inputs)

        # k, v: (B, N_heads, N_in, slot_dim // N_heads)
        k = self.to_k(inputs).reshape(B, N_in, N_heads, -1).transpose(1, 2)
        v = self.to_v(inputs).reshape(B, N_in, N_heads, -1).transpose(1, 2)

        for _ in range(self.iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            q = (
                self.to_q(slots).reshape(B, K, N_heads, -1).transpose(1, 2)
            )  # (B, N_heads, K, slot_D // N_heads)
            attn_logits = (
                torch.einsum('bhnd,bhkd->bhnk', k, q) * self.scale
            )  # (B, N_heads, N_in, K).
            attn_sftmx = attn_logits.softmax(dim=-1) + self.eps  # softmax across slots
            attn_norm = attn_sftmx / torch.sum(attn_sftmx, dim=-2, keepdim=True)  # slotwise normalization (sum=1)

            '''
            attn: (B, N_heads, N_in, K)
            v   : (B, N_heads, N_in, slot_D // N_heads)
            '''
            updates = torch.einsum(
                'bhnk,bhnd->bhkd', attn_norm, v
            )  # (B, N_heads, K, slot_D // N_heads)
            updates = updates.transpose(1, 2).reshape(B, K, -1)  # (B, K, slot_D)
            # sum(Value x slotwise attention weight)

            slots = self.gru(
                updates.reshape(-1, self.slot_dim), slots_prev.reshape(-1, self.slot_dim)
            )

            slots = slots.reshape(B, -1, self.slot_dim)
            slots = slots + self.mlp(self.norm_mlp(slots))


        attn_sftmx = torch.mean(attn_sftmx, dim=1) # (B, N_in, K)

        return attn_sftmx, slots


class EDSR(nn.Module):
    def __init__(self, args, conv=default_conv):
        super(EDSR, self).__init__()
        self.args = args
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale[0]
        act = nn.ReLU(True)
        url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        if url_name in url:
            self.url = url[url_name]
        else:
            self.url = None
        # self.sub_mean = MeanShift(args.rgb_range)
        # self.add_mean = MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)

        if args.no_upsampling:
            self.out_dim = n_feats + args.slot_dim
        else:
            self.out_dim = args.n_colors
            # define tail module
            
            self.tail = nn.Sequential(nn.Conv2d(n_feats + args.slot_dim, n_feats, 3, 1, 1),
                                                nn.LeakyReLU(inplace=True), 
                                                nn.Conv2d(n_feats, self.out_dim, 3, 1, 1))
            
        self.slot_attention = SlotAttention(args)

    def imresize(self, x, scale_factor=None, size=None):
        if type(self.args.upsample_mode) == str:
            if scale_factor == None:
                up_x = F.interpolate(x, 
                    size=size, 
                    mode=self.args.upsample_mode)
            else:
                up_x = F.interpolate(x, 
                    scale_factor=scale_factor, 
                    mode=self.args.upsample_mode)
        elif type(self.args.upsample_mode) == list:
            if scale_factor == None:
                tmp_res = F.interpolate(x, 
                    size=size, 
                    mode=self.args.upsample_mode[0])
            else:
                tmp_res = F.interpolate(x, 
                    scale_factor=scale_factor, 
                    mode=self.args.upsample_mode[0])
            for m in self.args.upsample_mode[1:]:
                if scale_factor == None:
                    tmp_res_ = F.interpolate(x, size=size, mode=m)
                    tmp_res = torch.cat([tmp_res, tmp_res_], dim=1)
                else:
                    tmp_res_ = F.interpolate(x, scale_factor=scale_factor, mode=m)
                    tmp_res = torch.cat([tmp_res, tmp_res_], dim=1)
            tmp_res = tmp_res / len(self.args.upsample_mode)
            up_x = tmp_res
        else:
            if scale_factor == None:
                up_x = F.interpolate(x, size=size, mode='bicubic')
            else:
                up_x = F.interpolate(x, scale_factor=scale_factor, mode='bicubic')

        return up_x

    def forward(self, x, scale_factor=None, size=None, mode='test'): 
        #x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        body_feat = res + x

        B,D,h,w = x.size()

        attn_sftmx, slots = self.slot_attention(body_feat.reshape(B, D, h*w).permute(0, 2, 1))
        # attn_sftmx (B, h*w, K)
        # slots (B, K, D`)

        slot_feat = torch.einsum('bnk,bkd->bnd', attn_sftmx, slots) # (B, h*w, D`)
        slot_feat = slot_feat.permute(0, 2, 1).reshape(B, -1, h, w) # (B, D`, h, w)

        x = torch.cat([body_feat, slot_feat], dim=1) # (B, D + D`, h, w)

        if self.args.no_upsampling:
            return x, attn_sftmx
        else:
            up_x = self.imresize(x=x,
                                 scale_factor=scale_factor,
                                 size=size)

            x = self.tail(up_x)
            return x

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


@register('edsr-tiny-sa2')
def make_edsr_tiny(n_resblocks=8, n_feats=16, res_scale=1, scale=2, 
                    no_upsampling=False, upsample_mode='bicubic',rgb_range=1,
                    slot_num=6, slot_iters=3, slot_attn_heads=1,
                    slot_dim=16, slot_mlp_hid_dim=16, slot_init_mode='learnable'):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.upsample_mode = upsample_mode

    args.rgb_range = rgb_range
    args.n_colors = 3

    args.slot_input_dim = n_feats
    args.slot_num = slot_num
    args.slot_iters = slot_iters
    args.slot_attn_heads = slot_attn_heads
    args.slot_dim = slot_dim
    args.slot_mlp_hid_dim = slot_mlp_hid_dim
    args.slot_init_mode = slot_init_mode

    return EDSR(args)

@register('edsr-light-sa2')
def make_edsr_light(n_resblocks=16, n_feats=32, res_scale=1, scale=2, 
                    no_upsampling=False, upsample_mode='bicubic',rgb_range=1,
                    slot_num=8, slot_iters=3, slot_attn_heads=1,
                    slot_dim=32, slot_mlp_hid_dim=32, slot_init_mode='learnable'):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.upsample_mode = upsample_mode

    args.rgb_range = rgb_range
    args.n_colors = 3

    args.slot_input_dim = n_feats
    args.slot_num = slot_num
    args.slot_iters = slot_iters
    args.slot_attn_heads = slot_attn_heads
    args.slot_dim = slot_dim
    args.slot_mlp_hid_dim = slot_mlp_hid_dim
    args.slot_init_mode = slot_init_mode

    return EDSR(args)


@register('edsr-baseline-sa2')
def make_edsr_baseline(n_resblocks=16, n_feats=64, res_scale=1, scale=2, 
                       no_upsampling=False, upsample_mode='bicubic',rgb_range=1,
                       slot_num=10, slot_iters=3, slot_attn_heads=2,
                       slot_dim=64, slot_mlp_hid_dim=64, slot_init_mode='learnable'):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.upsample_mode = upsample_mode

    args.rgb_range = rgb_range
    args.n_colors = 3

    args.slot_input_dim = n_feats
    args.slot_num = slot_num
    args.slot_iters = slot_iters
    args.slot_attn_heads = slot_attn_heads
    args.slot_dim = slot_dim
    args.slot_mlp_hid_dim = slot_mlp_hid_dim
    args.slot_init_mode = slot_init_mode

    return EDSR(args)


@register('edsr-sa2')
def make_edsr(n_resblocks=32, n_feats=256, res_scale=0.1, scale=2, 
              no_upsampling=False, upsample_mode='bicubic', rgb_range=1,
              slot_num=10, slot_iters=3, slot_attn_heads=4,
              slot_dim=128, slot_mlp_hid_dim=128, slot_init_mode='learnable'):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.upsample_mode = upsample_mode

    args.rgb_range = rgb_range
    args.n_colors = 3

    args.slot_input_dim = n_feats
    args.slot_num = slot_num
    args.slot_iters = slot_iters
    args.slot_attn_heads = slot_attn_heads
    args.slot_dim = slot_dim
    args.slot_mlp_hid_dim = slot_mlp_hid_dim
    args.slot_init_mode = slot_init_mode
    
    return EDSR(args)
