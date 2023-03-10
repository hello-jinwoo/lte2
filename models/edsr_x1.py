# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

# from bicubic_pytorch import core

from models import register


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


class ScaleModule(nn.Module):
    def __init__(self, embed_dim=96):
        super(ScaleModule, self).__init__()

        self.path1 = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),
                                                    nn.LeakyReLU(inplace=True), 
                                                    nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1))
        self.path2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x, scale_factor):

        _,_,h,w = x.size()
        up_x = core.imresize(x, scale=1/scale_factor)
        p1 = self.path1(up_x)
        p1 = core.imresize(p1, sizes=(h,w))

        p2 = self.path2(x)

        corr = p1*p2
        corr = torch.sigmoid(corr)

        x = x + corr*x
        return x


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
        m_body = nn.ModuleList([
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ])
        m_body.append(conv(n_feats, n_feats, kernel_size))

        self.head = nn.Sequential(*m_head)
        # self.body = nn.Sequential(*m_body)
        self.body = m_body

        if args.no_upsampling:
            self.out_dim = n_feats
        else:
            self.out_dim = args.n_colors
            # define tail module
            
            self.tail = nn.Sequential(nn.Conv2d(n_feats, n_feats, 3, 1, 1),
                                                    nn.LeakyReLU(inplace=True), 
                                                    nn.Conv2d(n_feats, self.out_dim, 3, 1, 1))
            

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
                    tmp_res = tmp_res + F.interpolate(x, 
                                                      size=size, 
                                                      mode=m)
                else:
                    tmp_res = tmp_res + F.interpolate(x, 
                                                      scale_factor=scale_factor, 
                                                      mode=m)
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
        _,_,h,w = x.size()

        x = self.head(x)

        # res = self.body(x) 
        reproduce_features = []
        res = x
        for i in range(len(self.body)):
            res = self.body[i](res)
            if i in self.args.reproduce_layers and mode == 'train':
                reproduce_features.append(res)
        res += x

        up_x = res
        if -1 in self.args.reproduce_layers and mode == 'train':
            reproduce_features.append(up_x)
        x = self.tail(up_x)

        if mode == 'train':
            return x, reproduce_features
        else:
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


@register('edsr-light-x1')
def make_edsr_baseline(n_resblocks=16, n_feats=32, res_scale=1, scale=2, 
                       no_upsampling=False, upsample_mode='bicubic',rgb_range=1,
                       reproduce_layers=[3,7,11,15,-1]):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.upsample_mode = upsample_mode

    args.rgb_range = rgb_range
    args.n_colors = 3

    args.reproduce_layers = reproduce_layers

    return EDSR(args)


@register('edsr-baseline-x1')
def make_edsr_baseline(n_resblocks=16, n_feats=64, res_scale=1, scale=2, 
                       no_upsampling=False, upsample_mode='bicubic',rgb_range=1,
                       reproduce_layers=[3,7,11,15,-1]):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.upsample_mode = upsample_mode

    args.rgb_range = rgb_range
    args.n_colors = 3

    args.reproduce_layers = reproduce_layers

    return EDSR(args)


@register('edsr-x1')
def make_edsr(n_resblocks=32, n_feats=256, res_scale=0.1, scale=2, 
              no_upsampling=False, upsample_mode='bicubic', rgb_range=1,
              reproduce_layers=[7,15,23,31,-1]):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.upsample_mode = upsample_mode

    args.rgb_range = rgb_range
    args.n_colors = 3

    args.reproduce_layers = reproduce_layers

    return EDSR(args)
