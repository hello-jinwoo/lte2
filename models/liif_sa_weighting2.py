import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord

# ensemble weights by similarity between slots 

@register('liif-sa-weighting2')
class LIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True, slot_dim=64):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.slot_dim = slot_dim

        self.encoder = models.make(encoder_spec)

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 9
                # slot_dim *= 9
            imnet_in_dim += 2 # attach coord
            if self.cell_decode:
                imnet_in_dim += 2
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None
        
        # n_ref = 1
        # if local_ensemble:
        #     slot_dim *= 4
        #     n_ref *= 4
        # self.slots_to_weights = nn.Sequential(nn.Linear(slot_dim, slot_dim),
        #                                       nn.ReLU(),
        #                                       nn.Linear(slot_dim, n_ref),
        #                                       nn.Softmax(dim=-1))

    def gen_feat(self, inp, scale_factor=None):
        if scale_factor != None:
            self.feat = self.encoder(inp, scale_factor=scale_factor)
        else:
            self.feat = self.encoder(inp)
        return self.feat

    def query_rgb(self, coord, cell=None):
        feat = self.feat
        slot_feat = feat[:, -self.slot_dim:, :, :]

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        if self.local_ensemble:
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
        slots = []
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
                q_slot_feat = F.grid_sample(
                    slot_feat, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                q_coord = F.grid_sample(
                    feat_coord, coord_.flip(-1).unsqueeze(1),
                    mode='nearest', align_corners=False)[:, :, 0, :] \
                    .permute(0, 2, 1)
                rel_coord = coord - q_coord
                rel_coord[:, :, 0] *= feat.shape[-2]
                rel_coord[:, :, 1] *= feat.shape[-1]
                inp = torch.cat([q_feat, rel_coord], dim=-1)
                slots.append(q_slot_feat)

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

        if self.local_ensemble:
            slots = torch.stack(slots, dim=-1) # (B, N, D_slots, 4)
            normed_slots = slots / torch.linalg.norm(slots, dim=-2, keepdim=True)
            slots_self_attn_map = normed_slots[:, :, :, None, :] * normed_slots[:, :, :, :, None] # (B, N, D_slots, 4, 4)
            slots_self_attn_map = torch.mean(slots_self_attn_map, dim=-3) # (B, N, 4, 4)
            diagonal_mask = 1 - torch.eye(slots_self_attn_map.shape[-1]).to(slots_self_attn_map.device) # (4, 4)
            ensemble_weights = slots_self_attn_map * diagonal_mask[None, None, :, :] # masking diagonal
            ensemble_weights = torch.mean(ensemble_weights, dim=-1) # (B, N, 4)
            ensemble_weights = ensemble_weights / torch.sum(ensemble_weights, dim=-1, keepdim=True)
            
            ret = 0
            for i, pred in enumerate(preds):
                # ret = ret + pred * (area / tot_area).unsqueeze(-1)
                ret = ret + pred * ensemble_weights[..., i: i+1]
        else:
            ret = preds[0]
            
        return ret

    def forward(self, inp, coord, cell, scale_factor=None):
        if scale_factor != None:
            self.gen_feat(inp, scale_factor=scale_factor)
        else:
            self.gen_feat(inp)
        return self.query_rgb(coord, cell)
