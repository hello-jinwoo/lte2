# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

# from bicubic_pytorch import core

from models import register


"""
ScaleAdaptiveLocalWindow network (SALWnet)
"""

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
        # self.body = nn.Sequential(*m_body)

        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors
            # define tail module
            
            hid_dim = len(self.args.upsample_mode) * n_feats if type(self.args.upsample_mode) == list else n_feats
            self.tail = nn.Sequential(nn.Conv2d(hid_dim, n_feats, 3, 1, 1),
                                      nn.LeakyReLU(inplace=True),
                                      nn.Conv2d(n_feats, n_feats, 1))

            mhsa_encoder_layer = nn.TransformerEncoderLayer(d_model=args.mhsa_dim, nhead=args.mhsa_head, batch_first=True)
            self.mhsa_after_tail = nn.TransformerEncoder(mhsa_encoder_layer, num_layers=args.mhsa_layer)

            self.final = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, 1, 1),
                                       nn.LeakyReLU(inplace=True),
                                       nn.Conv2d(n_feats, 3, 1))
            

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
        x = self.head(x)

        # res = self.body(x)
        # res += x

        B,D,h,w = x.size()

        up_x = self.imresize(x=x,
                             scale_factor=scale_factor,
                             size=size)
        _,_,H,W = up_x.size()
        # up_x = self.head(up_x)
        x = self.tail(up_x) # (B, D, H, W)

        

        # uws: upscale_window_size
        # lws: local_window_size (have to be multiple of uws)
        if scale_factor != None:
            uws = math.ceil(scale_factor)
        else:
            uws = math.ceil(H/h)
        lws = self.args.local_window_size
        if lws % uws != 0:
            if lws % uws > uws - lws % uws:
                lws = lws + (uws - lws % uws)
            else:
                lws = lws - (lws % uws)

        # pad input to the multiple of lws (H` and W` are multiple of lws)
        pad_size = [0, 0, 0, 0]
        if H % lws != 0:
            h_margin = lws * (H // lws + 1) - H
            pad_size[3] = h_margin
        if W % lws != 0:
            h_margin = lws * (W // lws + 1) - W
            pad_size[1] = h_margin
        pad_size = tuple(pad_size)
        padded_x = F.pad(input=x, pad=pad_size, mode='reflect') # (B, D, H`, W`)

        # uws
        uws_patches = padded_x.unfold(2, uws, uws).unfold(3, uws, uws) # (B, D, h_uws_patches, w_uws_patches, uws, uws) - (A)
        unfold_shape = uws_patches.size() # (B, D, h_uws_patches, w_uws_patches, uws, uws)
        _,_,h_uws_patches,w_uws_patches,_,_ = uws_patches.size()
        uws_patches = uws_patches.reshape(B, D, -1, uws*uws) # (B, D, N_uws_patches, uws*uws) - (B)
        uws_feat = uws_patches.permute(0, 2, 3, 1).reshape(-1, uws*uws, D) # (B * N_uws_patches, uws*uws, D) - (C)

        lws_patches = padded_x.unfold(2, lws, lws).unfold(3, lws, lws) # (B, D, h_lws_patches, w_lws_patches, lws, lws)
        lws_patches = torch.mean(lws_patches, dim=(-2, -1)) # (B, D, h_lws_patches, w_lws_patches)
        lws_patches = F.interpolate(lws_patches, scale_factor=lws//uws, mode='nearest') # (B, D, h_uws_patches, w_uws_patches)
        lws_patches = lws_patches.reshape(B, D, -1)[..., None] # (B, D, N_uws_patches, 1)
        lws_feat = lws_patches.permute(0, 2, 3, 1).reshape(-1, 1, D) # (B * N_uws_patches, 1, D)

        # global
        global_patches = torch.mean(x, dim=(2, 3), keepdim=True) # (B, D, 1, 1)
        global_patches = F.interpolate(global_patches, size=(h_uws_patches, w_uws_patches), mode='nearest') # (B, D, h_uws_patches, w_uws_patches)
        global_patches = global_patches.reshape(B, D, -1)[..., None] # (B, D, N_uws_patches, 1)
        global_feat = global_patches.permute(0, 2, 3, 1).reshape(-1, 1, D) # (B * N_uws_patches, 1, D)
        
        # MHSA
        fusion_feat = torch.cat([uws_feat, lws_feat, global_feat], dim=1) # (B * N_uws_patches, uws*uws+2, D)
        x_mhsa = self.mhsa_after_tail(fusion_feat)
        x_mhsa = x_mhsa[:, :-2, :] # cut out last two feat (lws, global) # (B * N_uws_patches, uws*uws, D) - (C`)
        x_mhsa = x_mhsa.reshape(B, -1, uws*uws, D).permute(0, 3, 1, 2) # (B, D, N_uws_patches, uws*uws) - (B`)
        x_mhsa = x_mhsa.reshape(B, D, h_uws_patches, w_uws_patches, uws, uws) # (B, D, h_uws_patches, w_uws_patches, uws, uws) - (A`)
        output_h = unfold_shape[2] * unfold_shape[4]
        output_w = unfold_shape[3] * unfold_shape[5]
        x_mhsa = x_mhsa.permute(0, 1, 2, 4, 3, 5).contiguous()
        x_mhsa = x_mhsa.view(B, D, output_h, output_w) # (B, D, H`, W`)
        x = x_mhsa[:, :, :H, :W] # cut out the padded part # (B, D, H, W)

        # final CNN (n_feats -> 3)
        x = self.final(x)

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


@register('salwnet1-tiny')
def make_edsr_tiny(n_resblocks=8, n_feats=16, res_scale=1, scale=2, 
                    no_upsampling=False, upsample_mode='bicubic',rgb_range=1,
                    local_window_size=24, mhsa_dim=16, mhsa_head=1, mhsa_layer=2):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.upsample_mode = upsample_mode

    args.rgb_range = rgb_range
    args.n_colors = 3
    args.local_window_size = local_window_size
    args.mhsa_dim = mhsa_dim
    args.mhsa_head = mhsa_head
    args.mhsa_layer = mhsa_layer
    return EDSR(args)

@register('salwnet1-light')
def make_edsr_light(n_resblocks=16, n_feats=32, res_scale=1, scale=2, 
                    no_upsampling=False, upsample_mode='bicubic',rgb_range=1,
                    local_window_size=24, mhsa_dim=32, mhsa_head=1, mhsa_layer=3):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.upsample_mode = upsample_mode

    args.rgb_range = rgb_range
    args.n_colors = 3
    args.local_window_size = local_window_size
    args.mhsa_dim = mhsa_dim
    args.mhsa_head = mhsa_head
    args.mhsa_layer = mhsa_layer
    return EDSR(args)


@register('salwnet1-baseline')
def make_edsr_baseline(n_resblocks=16, n_feats=64, res_scale=1, scale=2, 
                       no_upsampling=False, upsample_mode='bicubic',rgb_range=1,
                       local_window_size=24, mhsa_dim=64, mhsa_head=4, mhsa_layer=6):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.upsample_mode = upsample_mode

    args.rgb_range = rgb_range
    args.n_colors = 3
    args.local_window_size = local_window_size
    args.mhsa_dim = mhsa_dim
    args.mhsa_head = mhsa_head
    args.mhsa_layer = mhsa_layer
    return EDSR(args)


@register('salwnet1')
def make_edsr(n_resblocks=32, n_feats=256, res_scale=0.1, scale=2, 
              no_upsampling=False, upsample_mode='bicubic', rgb_range=1,
              local_window_size=24, mhsa_dim=128, mhsa_head=6, mhsa_layer=8):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.upsample_mode = upsample_mode

    args.rgb_range = rgb_range
    args.n_colors = 3
    args.local_window_size = local_window_size
    args.mhsa_dim = mhsa_dim
    args.mhsa_head = mhsa_head
    args.mhsa_layer = mhsa_layer
    return EDSR(args)
