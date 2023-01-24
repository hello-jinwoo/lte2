# modified from: https://github.com/thstkdgus35/EDSR-PyTorch

import math
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F

# from bicubic_pytorch import core

from models import register

"""
jw -> jw2 : ensembel feature upsample using concat 
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
        n_feats_target = args.n_feats_target
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
            
            self.tail = nn.Sequential(nn.Conv2d(n_feats_target, n_feats_target, 3, 1, 1),
                                                    nn.LeakyReLU(inplace=True), 
                                                    nn.Conv2d(n_feats_target, self.out_dim, 3, 1, 1))
        
        # TODO: naming?
        n_upsample = len(args.upsample_mode)
        self.channel_sync =  nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(n_feats * n_upsample, n_feats, 1, 1, 0),
                nn.ReLU(inplace=True),
                nn.Conv2d(n_feats, n_feats, 1, 1, 0)
            ) for _ in range(len(args.reproduce_layers))
        ])
        
        self.reproduce_networks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(n_feats * n_upsample, n_feats_target, 3, 1, 1), # 
                nn.ReLU(inplace=True),
                nn.Conv2d(n_feats_target, n_feats_target, 3, 1, 1)
            ) for _ in range(len(args.reproduce_layers))
        ])

        # self.scale_layers = nn.ModuleList([
        #     ScaleModule(n_feats) for _ in range(len(args.reproduce_layers))
        # ])

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
        _,_,h,w = x.size()

        x = self.head(x)

        # res = self.body(x) 
        reproduce_features = []
        res = x
        for i in range(len(self.body)):
            res = self.body[i](res)
            if i in self.args.reproduce_layers:
                # upsample
                up_res = self.imresize(x=res,
                                    scale_factor=scale_factor,
                                    size=size)
                
                net_idx = self.args.reproduce_layers.index(i)
                up_res = self.reproduce_networks[net_idx](up_res)

                if mode == 'train':
                    reproduce_features.append(up_res)

                # downsample
                if scale_factor == None:
                    new_res = self.imresize(x=res,
                                            size=(h, w))
                    new_res = self.channel_sync[net_idx](new_res)
                elif size == None:
                    d_scale_factor = 1 / scale_factor
                    new_res = self.imresize(x=res,
                                            scale_factor=1 / scale_factor)
                    new_res = self.channel_sync[net_idx](new_res)

                
                res += new_res

        res += x

        if self.args.no_upsampling:
            up_x = res
        else:
            # up_x = core.imresize(res, sizes=(round(h*scale_factor),round(w*scale_factor)))
            # up_x = F.interpolate(res, size=(round(h*scale_factor),round(w*scale_factor)), mode=self.args.upsample_mode)
            up_x = self.imresize(x=x, 
                                 scale_factor=scale_factor, 
                                 size=size)

            if -1 in self.args.reproduce_layers:
                up_x = self.reproduce_networks[-1](up_x)
                if mode == 'train':
                    reproduce_features.append(up_x)

            x = self.tail(up_x)
            # x = self.tail(res)
        #x = self.add_mean(x)

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


@register('edsr-baseline-jw2')
def make_edsr_baseline(n_resblocks=16, n_feats=48, n_feats_target=64, res_scale=1, scale=2, 
                       no_upsampling=False, upsample_mode='bicubic',rgb_range=1,
                       reproduce_layers=[3,7,11,15,-1]):
    args = Namespace()
    args.n_resblocks = n_resblocks
    args.n_feats = n_feats
    args.n_feats_target = n_feats_target
    args.res_scale = res_scale

    args.scale = [scale]
    args.no_upsampling = no_upsampling
    args.upsample_mode = upsample_mode

    args.rgb_range = rgb_range
    args.n_colors = 3

    args.reproduce_layers = reproduce_layers

    return EDSR(args)


@register('edsr-jw2')
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
