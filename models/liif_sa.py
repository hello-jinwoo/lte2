import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from utils import make_coord


@register('liif-sa')
class LIIF(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None,
                 local_ensemble=True, feat_unfold=True, cell_decode=True, attn_ensemble=True, n_refs=4, attn_ref_mode='random'):
        super().__init__()
        self.local_ensemble = local_ensemble
        self.feat_unfold = feat_unfold
        self.cell_decode = cell_decode
        self.attn_ensemble = attn_ensemble
        self.n_refs = n_refs
        self.attn_ref_mode = attn_ref_mode

        self.encoder = models.make(encoder_spec)

        if imnet_spec is not None:
            imnet_in_dim = self.encoder.out_dim
            if self.feat_unfold:
                imnet_in_dim *= 9
            imnet_in_dim += 2 # attach coord
            if self.cell_decode:
                imnet_in_dim += 2
            self.imnet = models.make(imnet_spec, args={'in_dim': imnet_in_dim})
        else:
            self.imnet = None

        # feat_dim = self.encoder.out_dim
        # if self.feat_unfold:
        #     feat_dim *= 9
        # self.ref_feat_prj = nn.Linear(feat_dim, feat_dim // self.n_refs)

    def gen_feat(self, inp):
        self.feat, self.attn_sftmx = self.encoder(inp)
        return self.feat, self.attn_sftmx

    def query_rgb(self, coord, cell=None):
        feat = self.feat # (B, D, h, W)
        B, D, h, w = feat.shape
        N = coord.shape[1]
        attn_sftmx = self.attn_sftmx # (B, h*w, K)
        ret = None

        if self.imnet is None:
            ret = F.grid_sample(feat, coord.flip(-1).unsqueeze(1),
                mode='nearest', align_corners=False)[:, :, 0, :] \
                .permute(0, 2, 1)
            return ret

        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])
            B, D, h, w = feat.shape

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        if self.attn_ensemble:
            pixel_coord = (coord + 1) / 2 # -1~1 -> 0~1
            pixel_coord [..., 0] *= h
            pixel_coord [..., 1] *= w
            pixel_coord = pixel_coord.long() # (B, N, 2)
            pixel_coord = torch.clip(pixel_coord, 0, w*h-1)
            pixel_coord_1d = pixel_coord[..., 0] * h + pixel_coord[..., 1] # (B, N)
            pixel_coord_1d_list = [(b, int(pixel_coord_1d[b][i])) for b in range(B) for i in range(N)]

            segm_idx_whole = torch.max(attn_sftmx, dim=2).indices # (B, h*w)
            segm_idx = segm_idx_whole[list(torch.tensor(pixel_coord_1d_list).T)].reshape(B, N) # (B, N)
            segm_idx_list = [(b, int(segm_idx[b][i])) for b in range(B) for i in range(N)]
            pixel_segm_prob_map = attn_sftmx.permute(0, 2, 1)[list(torch.tensor(segm_idx_list).T)] # (B*N, h*w)
            pixel_segm_prob_map = pixel_segm_prob_map.reshape(B, N, h*w) # (B, N, h*w)
            if self.attn_ref_mode == 'random':
                rand_prob = torch.rand_like(pixel_segm_prob_map)
                ref_coords = torch.topk(pixel_segm_prob_map + rand_prob, k=self.n_refs, dim=2).indices # (B, N, n_refs)
                ref_probs = torch.topk(pixel_segm_prob_map + rand_prob, k=self.n_refs, dim=2).values # (B, N, n_refs)
            elif self.attn_ref_mode == 'topk':
                ref_coords = torch.topk(pixel_segm_prob_map, k=self.n_refs, dim=2).indices # (B, N, n_refs)
                ref_probs = torch.topk(pixel_segm_prob_map, k=self.n_refs, dim=2).values # (B, N, n_refs)
            elif self.attn_ref_mode == 'random_dist_const':
                pass
            
            feat_reshaped = feat.reshape(B, D, -1).permute(0, 2, 1).reshape(-1, D) # (B*N, D)
            ref_feats = feat_reshaped[ref_coords.reshape(-1, self.n_refs)] # (B*N, n_refs, D)
            ref_feats = ref_feats.reshape(B, N, self.n_refs, D) # (B, N, n_refs, D)

            _ref_relative_coords = ref_coords - pixel_coord_1d[..., None].expand(B, N, self.n_refs)
            ref_relative_coords = torch.zeros(B, N, self.n_refs, 2) # (B, N, n_refs, 2)
            ref_relative_coords[..., 0] = (_ref_relative_coords // h) / h
            ref_relative_coords[..., 1] = (_ref_relative_coords % h) / w
            # ref_relative_coords = (ref_relative_coords - 0.5) * 2 # -1~1

            ref_feats = torch.cat([ref_feats, ref_relative_coords.to(ref_feats.device)], dim=-1) # (B, N, n_refs, D+2)
            if self.cell_decode:
                rel_cell = cell.clone()
                rel_cell[:, :, 0] *= feat.shape[-2]
                rel_cell[:, :, 1] *= feat.shape[-1]
                rel_cell = rel_cell[:, :, None, :].expand(B, N, self.n_refs, -1)
                ref_feats = torch.cat([ref_feats, rel_cell], dim=-1) # (B, N, n_refs, D+4)

            ref_probs /= torch.sum(ref_probs, dim=-1, keepdim=True) # normalize (B, N, n_refs)
            ret = 0
            for i in range(self.n_refs):
                inp = ref_feats[:, :, i, :] # (B, N, n_refs, D`) -> (B, N, D`)
                pred = self.imnet(inp.view(B*N, -1)).view(B, N, -1) # (B, N, D`)
                ret += pred * ref_probs[..., i:i+1] # (B, N, D`)

        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6

            preds = []
            areas = []
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
                    q_coord = F.grid_sample(
                        feat_coord, coord_.flip(-1).unsqueeze(1),
                        mode='nearest', align_corners=False)[:, :, 0, :] \
                        .permute(0, 2, 1)
                    rel_coord = coord - q_coord
                    rel_coord[:, :, 0] *= feat.shape[-2]
                    rel_coord[:, :, 1] *= feat.shape[-1]
                    inp = torch.cat([q_feat, rel_coord], dim=-1)

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
            
            if ret == None:
                ret = 0
                for pred, area in zip(preds, areas):
                    ret = ret + pred * (area / tot_area).unsqueeze(-1)
            else:
                ret_local = 0
                for pred, area in zip(preds, areas):
                    ret_local = ret_local + pred * (area / tot_area).unsqueeze(-1)
                total_refs = self.n_refs + 4
                ret = ret * (self.n_refs) / total_refs + ret_local * 4 / total_refs
        
        return ret

    def forward(self, inp, coord, cell):
        self.gen_feat(inp)
        return self.query_rgb(coord, cell)
