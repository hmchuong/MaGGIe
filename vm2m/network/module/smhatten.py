'''
Source: https://github.com/lim142857/Sparsifiner/blob/main/src/models/sparsifiner.py
'''
import math
import torch
from torch import nn
from timm.models.layers import trunc_normal_

def compute_sparsity(x):
    total_num = torch.numel(x)
    num_non_zero = torch.count_nonzero(x)
    num_zero = total_num - num_non_zero
    sparsity = num_zero / total_num
    return sparsity

class MaskPredictor(nn.Module):
    """ Mask Predictor using Low rank MHA"""

    def __init__(self,
                 dim,
                 num_heads=8,
                 num_tokens=197,
                 attn_keep_rate=0.25,
                 reduce_n_factor=8,
                 reduce_c_factor=2,
                 share_inout_proj=False,
                 qk_scale=None
                 ):
        super().__init__()
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.reduced_c = self.head_dim // reduce_c_factor
        self.reduced_n = int(num_tokens // reduce_n_factor)
        self.scale = qk_scale or self.num_heads ** -0.5
        self.proj_c_q = nn.Linear(self.head_dim, self.reduced_c)
        self.proj_c_k = nn.Linear(self.head_dim, self.reduced_c)

        self.proj_n = nn.Parameter(torch.zeros(self.num_tokens, self.reduced_n))
        # trunc_normal_(self.proj_back_n, std=.02, a=0.)
        trunc_normal_(self.proj_n, std=.02)
        if share_inout_proj:
            self.proj_back_n = self.proj_n
        else:
            self.proj_back_n = nn.Parameter(torch.zeros(self.num_tokens, self.reduced_n))
            trunc_normal_(self.proj_back_n, std=.02)

        self.basis_threshold = nn.Threshold(cfg.SPAR.BASIS_THRESHOLD, 0.)
        self.basis_coef_threshold = nn.Threshold(cfg.SPAR.BASIS_COEF.THRESHOLD, 0.)

        self.attn_budget = math.ceil(attn_keep_rate * num_tokens)

    def forward(self, q, k, token_mask=None):
        # TODO: Perform full self-attention if attn_budget > token_budget
        if token_mask is not None:
            token_budget = token_mask[0].sum(dim=-1)
            self.attn_budget = token_budget if token_budget < self.attn_budget else self.attn_budget

        out_dict = {}
        cfg = self.cfg.SPAR

        B, H, N, C = q.shape
        assert self.num_tokens == N
        q, k = self.proj_c_q(q), self.proj_c_k(k)  # [B, H, N, c]
        if token_mask is not None:
            # token_mask: [B, N-1]
            q[..., 1:, :] = q[..., 1:, :].masked_fill(~token_mask[:, None, :, None], 0.)
            k[..., 1:, :] = k[..., 1:, :].masked_fill(~token_mask[:, None, :, None], 0.)

        k = k.permute(0, 1, 3, 2)  # [B, H, c, N]
        k = k @ self.proj_n  # [B, H, c, k]

        # TODO: should call this only once during inference.
        if self.training and self.cfg.LOSS.USE_ATTN_RECON:
            basis = self.proj_back_n.permute(1, 0)
        else:
            basis = self.proj_back_n.permute(1, 0)
            # basis[basis.abs() <= cfg.BASIS_THRESHOLD] = 0.
            # For Linear attention visualization
            basis = self.basis_threshold(basis.abs())

        # Compute low-rank approximation of the attention matrix
        # q: [B, H, N, c]   k: [B, H, c, K]
        cheap_attn = (q @ k) * self.scale  # [B, H, N, K]
        cheap_attn = cheap_attn[..., 1:, :]  # [B, H, N-1, K] remove cls token
        basis_coef = cheap_attn.softmax(dim=-1)  # [B, H, N-1, K] coef is naturally sparse
        if self.training and self.cfg.LOSS.USE_ATTN_RECON:
            approx_attn = basis_coef @ basis  # [B, H, N-1, N]
        else:
            if cfg.BASIS_COEF.USE_TOPK:
                basis_coef_topk, basis_coef_topk_indices = basis_coef.topk(cfg.BASIS_COEF.TOPK, sorted=False)
                basis_coef = torch.zeros_like(basis_coef, device=basis_coef.device)
                basis_coef.scatter_(-1, basis_coef_topk_indices, basis_coef_topk)
            elif cfg.BASIS_COEF.THRESHOLD > 0:
                # basis_coef[basis_coef <= cfg.BASIS_COEF.THRESHOLD] = 0.
                basis_coef = self.basis_coef_threshold(basis_coef)
            approx_attn = basis_coef @ basis  # [B, H, N-1, N]

        # Zero out attention connectivity columns corresponding to inactive tokens
        attn_score = approx_attn.clone()  # [B, H, N-1, N]
        if token_mask is not None:
            attn_score[..., 1:].masked_fill_(~token_mask[:, None, None, :], float('-inf'))  # [B, H, N-1, N]

        # Generate columns of instance dependent sparse attention connectivity pattern
        if cfg.ATTN_SCORE.USE_TOPK:
            # Top-k attention connectivity
            topk_cont_indices = torch.topk(attn_score, self.attn_budget, sorted=False)[1]  # [B, H, N-1, num_cont]
            attn_mask = torch.zeros_like(attn_score, dtype=attn_score.dtype, device=attn_score.device)
            attn_mask.scatter_(-1, topk_cont_indices, True)  # [B, H, N-1, N]
        elif cfg.ATTN_SCORE.THRESHOLD > 0:
            # Threshold attention connectivity
            attn_mask = torch.where(attn_score <= cfg.ATTN_SCORE.THRESHOLD, 0., 1.)
        else:
            raise NotImplementedError

        # Zero out attention connectivity rows corresponding to inactive tokens
        if token_mask is not None and cfg.PRUNE_ATTN_MATRIX_ROW:
            attn_mask *= token_mask[:, None, :, None]  # [B, H, N-1, N]

        # Add cls token back to attn mask
        cls_mask = torch.ones(B, H, 1, N, dtype=attn_mask.dtype, device=attn_mask.device)
        attn_mask = torch.cat([cls_mask, attn_mask], dim=2)  # [B, H, N, N]
        attn_mask.detach_()  # TODO: No gradient for attn_mask

        out_dict['basis_coef'] = basis_coef
        out_dict['approx_attn'] = approx_attn
        out_dict['attn_mask'] = attn_mask
        if not self.training:
            if cfg.OUT_BASIS_SPARSITY:
                out_dict['basis_sparsity'] = compute_sparsity(basis)
            if cfg.OUT_BASIS_COEF_SPARSITY:
                out_dict['basis_coef_sparsity'] = compute_sparsity(basis_coef)
            if cfg.OUT_ATTN_MASK_SPARSITY:
                out_dict['attn_mask_sparsity'] = compute_sparsity(attn_mask)
        return out_dict
    
class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_tokens=197,
            num_heads=8,
            attn_keep_rate=0.25,
            token_keep_rate=0.50,
            token_pruning_this_layer=False,
            reduce_n_factor=8,
            reduce_c_factor=2,
            share_inout_proj=False,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            cfg=None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.mask_predictor = MaskPredictor(
            dim,
            num_heads=num_heads,
            num_tokens=num_tokens,
            attn_keep_rate=attn_keep_rate,
            reduce_n_factor=reduce_n_factor,
            reduce_c_factor=reduce_c_factor,
            share_inout_proj=share_inout_proj,
            cfg=cfg,
        )
        self.token_pruning_this_layer = token_pruning_this_layer
        self.token_keep_rate = token_keep_rate
        self.token_budget = math.ceil(token_keep_rate * (num_tokens - 1))
        self.cfg = cfg

    def softmax_with_policy(self, attn, policy, eps=1e-6):
        # https://discuss.pytorch.org/t/how-to-implement-the-exactly-same-softmax-as-f-softmax-by-pytorch/44263/9
        B, H, N, N = attn.size()
        attn_policy = policy
        max_att = torch.max(attn, dim=-1, keepdim=True)[0]
        attn = attn - max_att
        # attn = attn.exp_() * attn_policy
        # return attn / attn.sum(dim=-1, keepdim=True)

        # for stable training
        attn = attn.to(torch.float32).exp_() * attn_policy.to(torch.float32)
        attn = (attn + eps / N) / (attn.sum(dim=-1, keepdim=True) + eps)
        return attn.type_as(max_att)

    def forward(self, x, prev_token_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q, k, v shape of [B, H, N, C]

        # Zero out key query values corresponding to inactive tokens
        if prev_token_mask is not None and not self.cfg.LOSS.USE_ATTN_RECON:
            q[..., 1:, :] = q[..., 1:, :].masked_fill(~prev_token_mask[:, None, :, None], 0.)
            k[..., 1:, :] = k[..., 1:, :].masked_fill(~prev_token_mask[:, None, :, None], 0.)
            v[..., 1:, :] = v[..., 1:, :].masked_fill(~prev_token_mask[:, None, :, None], 0.)

        out_dict = self.mask_predictor(q, k, prev_token_mask)
        attn_mask = out_dict['attn_mask']

        attn = (q @ k.transpose(-2, -1)) * self.scale
        unmasked_attn = attn.clone().softmax(dim=-1)
        if self.training and self.cfg.LOSS.USE_ATTN_RECON:
            attn = attn.softmax(dim=-1)  # Don't distort the token value when reconstructing attention
        else:
            # attn = self.softmax_with_policy(attn, attn_mask)
            attn.masked_fill_(~attn_mask.bool(), float('-inf'))
            attn = attn.softmax(dim=-1)
            attn = torch.nan_to_num(attn)  # Some rows are pruned and filled with '-inf'

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        out_dict['token_mask'] = prev_token_mask
        if self.token_pruning_this_layer and not self.cfg.LOSS.USE_ATTN_RECON:  # TODO: refactor this
            cls_attn = attn[:, :, 0, 1:]  # [B, H, N-1]
            token_score = cls_attn.mean(dim=1)  # [B, N-1]
            if prev_token_mask is not None:
                token_score = token_score.masked_fill(~prev_token_mask, float('-inf'))
            topk_token_indices = torch.topk(token_score, self.token_budget, sorted=False)[1]  # [B, left_tokens]
            new_token_mask = torch.zeros_like(token_score, dtype=torch.bool, device=token_score.device)
            new_token_mask.scatter_(-1, topk_token_indices, True)  # [B, N-1]
            out_dict['token_mask'] = new_token_mask  # TODO: would masked_fill be faster than indices fill?

        new_val = {'x': x, 'masked_attn': attn, 'unmasked_attn': unmasked_attn}
        out_dict.update(new_val)
        return out_dict