import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import models
from models import register
from utils import make_coord


@register('liif-patch2')
class LIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True,
                 patch_length=3, reference_patch_length=3):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.patch_length = patch_length
        self.patch_center_idx = patch_length ** 2 // 2
        self.reference_patch_length = reference_patch_length

        self.encoder = models.make(encoder_spec)

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 9
            # imnet_in_dim += 2 * patch_length ** 2 # attach coord
            self.pos_dim = imnet_in_dim // patch_length ** 2
            if self.cell_decode:
                imnet_in_dim += 2
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
            self.imnet_in_dim = imnet_in_dim
            
        else:
            self.imnet = None
        

    def pos_enc_sinu_2d(self, d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1) / width
        pos_h = torch.arange(0., height).unsqueeze(1) / height
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe
    
    def pos_enc_sinu_2d_for_point(self, pos, d_model):
        """
        :param d_model: dimension of the model
        :param pos: (B, 2) tensor
        :return: (B, d_model) tensor
        """
        B = pos.shape[0]
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(B, d_model).to(pos.device)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        div_term = div_term.to(pos.device)
        
        pe[:, 0:d_model:2] = torch.sin(pos[:, 0:1] * div_term)
        pe[:, 1:d_model:2] = torch.cos(pos[:, 0:1] * div_term)
        pe[:, d_model::2] = torch.sin(pos[:, 1:2] * div_term)
        pe[:, d_model + 1::2] = torch.cos(pos[:, 1:2] * div_term)

        return pe

    def gen_feat(self, inp, scale_factor=None):
        if scale_factor != None:
            self.feat = self.encoder(inp, scale_factor=scale_factor)
        else:
            self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        # coord = patched_coord[..., self.patch_center_idx]

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        # if self.local_ensemble:
        vx_lst = []
        vy_lst = []
        for i in range(-self.reference_patch_length, self.reference_patch_length + 1):
            if i != 0:
                vx_lst.append(i)
                vy_lst.append(i)
        eps_shift = 1e-6
        # else:
            # vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        preds = []
        areas = []
        for vx in vx_lst:
            for vy in vy_lst:
                coord_ = coord.clone()
                coord_[..., [i for i in range(0, coord.shape[-1], 2)]] += vx * rx + eps_shift
                coord_[..., [i for i in range(1, coord.shape[-1], 2)]] += vy * ry + eps_shift
                center_coord = coord_[..., self.patch_center_idx*2: (self.patch_center_idx+1)*2]
                center_coord.clamp_(-1 + 1e-6, 1 - 1e-6)
                q_feat = F.grid_sample(
                    feat, center_coord.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, center_coord.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord.repeat(1, 1, self.patch_length**2)
                rel_coord[:, :, [i for i in range(0, coord.shape[-1], 2)]] *= feat.shape[-2]
                rel_coord[:, :, [i for i in range(0, coord.shape[-1], 2)]] *= feat.shape[-1]
                rel_coord_sinu2d = torch.zeros(*rel_coord.shape[:2], self.pos_dim * self.patch_length ** 2)
                for i in range(0, coord.shape[-1], 2):
                    rel_coord_sinu2d[:, :, (i//2)*self.pos_dim: (i//2+1)*self.pos_dim] = self.pos_enc_sinu_2d_for_point(rel_coord[:, :, i:i+2].reshape(-1, 2), self.pos_dim).reshape(*rel_coord.shape[:2], self.pos_dim)
                # inp = torch.cat([q_feat, rel_coord], dim=-1)
                inp = q_feat + rel_coord_sinu2d.to(q_feat.device)

                if self.cell_decode:
                    rel_cell = cell.clone()
                    rel_cell[:, :, 0] *= feat.shape[-2]
                    rel_cell[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([inp, rel_cell], dim=-1)

                bs, q = coord.shape[:2]
                pred = self.imnet(inp.view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t
        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        return ret

    def forward(self, inp, coord, cell, scale_factor=None):
        if scale_factor != None:
            self.gen_feat(inp, scale_factor=scale_factor)
        else:
            self.gen_feat(inp)
        return self.query_rgb(coord, cell)
