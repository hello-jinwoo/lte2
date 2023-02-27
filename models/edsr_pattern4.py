# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

# from bicubic_pytorch import core

from models import register

'''
v4 : decoder as original edsr
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
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors
            # define tail module
            n_patterns = len(args.pattern_types) * 2
            self.tail = nn.Sequential(nn.Conv2d(n_feats + n_patterns, n_feats, 1),
                                      nn.LeakyReLU(inplace=True), 
                                      nn.Conv2d(n_feats, self.out_dim, 1))
            
    
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

    def make_patterns(self, h, w, device):
        max_pattern_size = self.args.max_pattern_size
        h_ = max_pattern_size * (h // max_pattern_size)
        w_ = max_pattern_size * (w // max_pattern_size)
        
        pattern_list = []

        # pattern from config
        for pattern_type in self.args.pattern_types:
            if pattern_type == 'vertical_1':
                pattern_list.append(torch.tile(torch.tensor([[1, -1]]), (h_, w_ // 2))) # (h_, w_)
                pattern_list.append(torch.tile(torch.tensor([[-1, 1]]), (h_, w_ // 2))) # (h_, w_)
            elif pattern_type == 'vertical_2':
                pattern_list.append(torch.tile(torch.tensor([[1, 1, -1, -1]]), (h_, w_ // 4))) # (h_, w_)
                pattern_list.append(torch.tile(torch.tensor([[-1, -1, 1, 1]]), (h_, w_ // 4))) # (h_, w_)
            elif pattern_type == 'vertical_4':
                pattern_list.append(torch.tile(torch.tensor([[1, 1, 1, 1, -1, -1, -1, -1]]), (h_, w_ // 8))) # (h_, w_)
                pattern_list.append(torch.tile(torch.tensor([[-1, -1, -1, -1, 1, 1, 1, 1]]), (h_, w_ // 8))) # (h_, w_)
            elif pattern_type == 'horizontal_1':
                pattern_list.append(torch.tile(torch.tensor([[1], [-1]]), (h_ // 2, w_))) # (h_, w_)
                pattern_list.append(torch.tile(torch.tensor([[-1], [1]]), (h_ // 2, w_))) # (h_, w_)
            elif pattern_type == 'horizontal_2':
                pattern_list.append(torch.tile(torch.tensor([[1], [1], [-1], [-1]]), (h_ // 4, w_))) # (h_, w_)
                pattern_list.append(torch.tile(torch.tensor([[-1], [-1], [1], [1]]), (h_ // 4, w_))) # (h_, w_)
            elif pattern_type == 'horizontal_4':
                pattern_list.append(torch.tile(torch.tensor([[1], [1], [1], [1], [-1], [-1], [-1], [-1]]), (h_ // 8, w_))) # (h_, w_)
                pattern_list.append(torch.tile(torch.tensor([[-1], [-1], [-1], [-1], [1], [1], [1], [1]]), (h_ // 8, w_))) # (h_, w_)
            elif pattern_type == 'diagonal_1':
                pattern_list.append(torch.tile(torch.tensor([[1, -1], 
                                                             [-1, 1]]), (h_ // 2, w_ // 2))) # (h_, w_)
                pattern_list.append(torch.tile(torch.tensor([[-1, 1], 
                                                             [1, -1]]), (h_ // 2, w_ // 2))) # (h_, w_)
            elif pattern_type == 'diagonal_2':
                pattern_list.append(torch.tile(torch.tensor([[1, 1, -1, -1],
                                                             [1, 1, -1, -1],
                                                             [-1, -1, 1, 1], 
                                                             [-1, -1, 1, 1]]), (h_ // 4, w_ // 4))) # (h_, w_)
                pattern_list.append(torch.tile(torch.tensor([[-1, -1, 1, 1],
                                                             [-1, -1, 1, 1],
                                                             [1, 1, -1, -1],
                                                             [1, 1, -1, -1]]), (h_ // 4, w_ // 4))) # (h_, w_)
            elif pattern_type == 'diagonal_4':
                pattern_list.append(torch.tile(torch.tensor([[1, 1, 1, 1, -1, -1, -1, -1],
                                                             [1, 1, 1, 1, -1, -1, -1, -1],
                                                             [1, 1, 1, 1, -1, -1, -1, -1],
                                                             [1, 1, 1, 1, -1, -1, -1, -1],
                                                             [-1, -1, -1, -1, 1, 1, 1, 1],
                                                             [-1, -1, -1, -1, 1, 1, 1, 1],
                                                             [-1, -1, -1, -1, 1, 1, 1, 1],
                                                             [-1, -1, -1, -1, 1, 1, 1, 1]]), (h_ // 8, w_ // 8))) # (h_, w_)
                pattern_list.append(torch.tile(torch.tensor([[-1, -1, -1, -1, 1, 1, 1, 1],
                                                             [-1, -1, -1, -1, 1, 1, 1, 1],
                                                             [-1, -1, -1, -1, 1, 1, 1, 1],
                                                             [-1, -1, -1, -1, 1, 1, 1, 1],
                                                             [1, 1, 1, 1, -1, -1, -1, -1],
                                                             [1, 1, 1, 1, -1, -1, -1, -1],
                                                             [1, 1, 1, 1, -1, -1, -1, -1],
                                                             [1, 1, 1, 1, -1, -1, -1, -1]]), (h_ // 8, w_ // 8))) # (h_, w_)
        pattern = torch.stack(pattern_list, dim=0)[None, ...].to(device).float() # (1, N, h_, w_)
        pattern = F.pad(input=pattern, pad=(0, 0, h - h_, w - w_), mode='replicate') # (1, N, h, w)

        return pattern

    def forward(self, x, scale_factor=None, size=None, mode='test'): 
        #x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        _,_,h,w = x.size()

        pattern = self.make_patterns(h=h, w=w, device=res.device) # (1, n_of_patterns, h, w)
        res = torch.cat([res, pattern.expand(res.shape[0], *pattern.shape[1:])], dim=1) 

        if self.args.no_upsampling:
            up_x = res
            return up_x
        else:
            up_x = self.imresize(x=res,
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


@register('edsr-tiny-pattern4')
def make_edsr_tiny(n_resblocks=8, n_feats=16, res_scale=1, scale=2, 
                    no_upsampling=False, upsample_mode='bicubic',rgb_range=1,
                    pattern_types=[], max_pattern_size=4):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.upsample_mode = upsample_mode

    args.rgb_range = rgb_range
    args.n_colors = 3

    args.pattern_types = pattern_types
    args.max_pattern_size = max_pattern_size
    return EDSR(args)

@register('edsr-light-pattern4')
def make_edsr_light(n_resblocks=16, n_feats=32, res_scale=1, scale=2, 
                    no_upsampling=False, upsample_mode='bicubic',rgb_range=1,
                    pattern_types=[], max_pattern_size=4):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.upsample_mode = upsample_mode

    args.rgb_range = rgb_range
    args.n_colors = 3

    args.pattern_types = pattern_types
    args.max_pattern_size = max_pattern_size
    return EDSR(args)


@register('edsr-baseline-pattern4')
def make_edsr_baseline(n_resblocks=16, n_feats=64, res_scale=1, scale=2, 
                       no_upsampling=False, upsample_mode='bicubic',rgb_range=1,
                       pattern_types=[], max_pattern_size=4):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.upsample_mode = upsample_mode

    args.rgb_range = rgb_range
    args.n_colors = 3

    args.pattern_types = pattern_types
    args.max_pattern_size = max_pattern_size
    return EDSR(args)


@register('edsr-pattern4')
def make_edsr(n_resblocks=32, n_feats=256, res_scale=0.1, scale=2, 
              no_upsampling=False, upsample_mode='bicubic', rgb_range=1,
              pattern_types=[], max_pattern_size=4):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.upsample_mode = upsample_mode

    args.rgb_range = rgb_range
    args.n_colors = 3

    args.pattern_types = pattern_types
    args.max_pattern_size = max_pattern_size
    return EDSR(args)
