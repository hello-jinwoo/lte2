import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

'''
geo v1 -> v2: geometric representation (8d to feat_dim encoder)
'''


@register('liif-geo2')
class LIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True, geo_ensemble=True):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.geo_ensemble = True

        self.encoder = models.make(encoder_spec)

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 9
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

        q_feat_dim = self.encoder.out_dim
        if self.feat_unfold:
            q_feat_dim *= 9
        self.q_feat_proj = nn.Linear(q_feat_dim, q_feat_dim // 4)

        geo_in_dim = 8
        if self.cell_decode:
            geo_in_dim += 2
        self.geo_encoder = nn.Sequential(nn.Linear(geo_in_dim, imnet_in_dim // 4),
                                         nn.ReLU(True),
                                         nn.Linear(imnet_in_dim // 4, imnet_in_dim))

    def gen_feat(self, inp):
        self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble or self.geo_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        inps = []
        rel_coords = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[:, :, 0] += vx * rx + eps_shift
                coord_[:, :, 1] += vy * ry + eps_shift
                coord_.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                if self.geo_ensemble:
                    q_feat = self.q_feat_proj(q_feat)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                if self.geo_ensemble:
                    inp = q_feat
                else:
                    inp = torch.cat([q_feat, rel_coord], dim=-1)

                if self.geo_ensemble:
                    inps.append(inp)
                    rel_coords.append(rel_coord)
                else:
                    bs, q = coord.shape[:2]
                    pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                    preds.append(pred)

                    area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                    areas.append(area + 1e-9)

        if self.geo_ensemble:
            inp = torch.cat(inps, dim=-1)
            rel_coord = torch.cat(rel_coords, dim=-1)
            if self.cell_decode:
                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]
                rel_coord = torch.cat([rel_coord, rel_cell], dim=-1)
            geo_feat = self.geo_encoder(rel_coord)
            inp = inp + geo_feat
            bs, q = inp.shape[:2]
            ret = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
        else:
            tot_area = torch.stack(areas).sum(dim=0)
            if self.local_ensemble:
                t = areas[0]; areas[0] = areas[3]; areas[3] = t
                t = areas[1]; areas[1] = areas[2]; areas[2] = t
            ret = 0
            for pred, area in zip(preds, areas):
                ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
