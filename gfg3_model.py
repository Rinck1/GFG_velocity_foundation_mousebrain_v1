#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GFG-v3: 保留原版 GFG 核心结构的 brain foundation model 扩容版。

不变量 (规格书一):
  1. 输入 = 基因空间 [u, s]，逐基因 token。
  2. token 特征 ⊇ (u, s)。
  3. gene encoder/decoder 参数跨基因共享 (per-gene MLP + set attention)。
  4. 双分支: manifold/state encoder + velocity/tangent encoder。
  5. 双独立 SoftVQ codebook; --no-vq 消融; 不删 VQ。
  6. 共享逐基因 state decoder → (u_hat, s_hat)。
  7. 观测空间速度 = J_D(z_s)·z_v (训练与推理同一算子)。
  9. 基因重排序 → 输出同样重排序 (set 架构天然置换等变, 无绝对位置编码)。

扩容: per-gene hidden / d_model / ISAB blocks / inducing / decoder 容量。
      original ~2M / medium ~15-30M / xlarge ~100-130M。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ------------------------------------------------------------------ spec
@dataclass(frozen=True)
class GFG3Spec:
    d_model: int = 384
    num_heads: int = 8
    num_blocks: int = 2          # ISAB block 数
    num_inducing: int = 128
    ff_mult: int = 2
    dropout: float = 0.10
    gene_mlp_hidden: tuple[int, ...] = (256, 256)   # per-gene tokenizer MLP
    decoder_hidden: tuple[int, ...] = (768, 384)    # shared per-gene decoder
    K: int = 32                  # 每个码本 code 数 (原版 32)
    vq_tau: float = 1.0
    vq_beta: float = 1.0         # commitment
    vq_gamma: float = 1.0        # entropy
    vq_ema_decay: float | None = None   # None = 梯度式 (原版); float = EMA
    detach_state_for_velocity: bool = True

    def validate(self):
        assert self.d_model % self.num_heads == 0
        assert self.num_blocks > 0
        return self


CONFIGS = {
    # original: 对齐原版 GFG 规模 (实测 1.53M)
    "original": GFG3Spec(d_model=192, num_heads=6, num_blocks=1, num_inducing=48,
                         ff_mult=2, gene_mlp_hidden=(96, 96),
                         decoder_hidden=(512, 256), K=32),
    # medium: 实测 18.88M (目标 15-30M)
    "medium": GFG3Spec(d_model=512, num_heads=8, num_blocks=2, num_inducing=192,
                       ff_mult=2, gene_mlp_hidden=(256, 256),
                       decoder_hidden=(1024, 512), K=64),
    # xlarge: 实测 117.90M (目标 100-130M)
    "xlarge": GFG3Spec(d_model=832, num_heads=13, num_blocks=4, num_inducing=320,
                       ff_mult=3, gene_mlp_hidden=(512, 512),
                       decoder_hidden=(1664, 832), K=128),
}


# ------------------------------------------------------------- set blocks
class MAB(nn.Module):
    """Set Transformer MAB (pre-norm)."""
    def __init__(self, d, heads, ff_mult, dropout):
        super().__init__()
        self.qn = nn.LayerNorm(d); self.kn = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.pn = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, d * ff_mult), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(d * ff_mult, d),
                                nn.Dropout(dropout))
        self.on = nn.LayerNorm(d)

    def forward(self, q, kv):
        h = q + self.attn(self.qn(q), self.kn(kv), self.kn(kv),
                          need_weights=False)[0]
        h = self.pn(h)
        return self.on(h + self.ff(h))


class ISAB(nn.Module):
    """Induced Set Attention Block: O(G·m)。"""
    def __init__(self, d, heads, m, ff_mult, dropout):
        super().__init__()
        self.inducing = nn.Parameter(torch.empty(1, m, d))
        nn.init.trunc_normal_(self.inducing, std=0.02)
        self.ab1 = MAB(d, heads, ff_mult, dropout)   # inducing ← set
        self.ab2 = MAB(d, heads, ff_mult, dropout)   # set ← inducing

    def forward(self, x):
        h = self.ab1(self.inducing.expand(x.shape[0], -1, -1), x)
        return self.ab2(x, h)


class SetCore(nn.Module):
    def __init__(self, spec: GFG3Spec):
        super().__init__()
        self.blocks = nn.ModuleList([
            ISAB(spec.d_model, spec.num_heads, spec.num_inducing,
                 spec.ff_mult, spec.dropout) for _ in range(spec.num_blocks)])
        self.norm = nn.LayerNorm(spec.d_model)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return self.norm(x)


def gene_mlp(d_in, hidden, d_out):
    L, w = [], d_in
    for h in hidden:
        L += [nn.Linear(w, h), nn.GELU()]; w = h
    L += [nn.Linear(w, d_out)]
    return nn.Sequential(*L)


# ------------------------------------------------------------------ VQ
class SoftVQ(nn.Module):
    """原版 SoftVectorQuantizer + EMA 可选 + DDP 同步 + 丰富监控。

    z: (B, G, D) 逐基因量化。梯度式(默认, DDP 天然安全) 或 EMA(同步充分统计量)。
    """
    def __init__(self, K, D, beta=1.0, gamma=1.0, tau=1.0,
                 ema_decay: float | None = None):
        super().__init__()
        self.K, self.D, self.beta, self.gamma, self.tau = K, D, beta, gamma, tau
        self.ema_decay = ema_decay
        self.normalize = True     # 余弦度量: 修复 LN 等半径球导致的均匀后验塌缩
        self.log_tau = nn.Parameter(torch.tensor(math.log(max((D ** 0.5) / 4, 1e-3))))
        self.embedding = nn.Parameter(torch.randn(K, D) * (1.0 / math.sqrt(D)))
        if ema_decay is not None:
            self.embedding.requires_grad_(False)
            self.register_buffer("_cluster_n", torch.zeros(K))
            self.register_buffer("_embed_sum", torch.zeros(K, D))
        self.register_buffer("usage_ema", torch.ones(K) / K)

    def forward(self, z):
        B, G, D = z.shape
        zf = z.reshape(-1, D)
        E = self.embedding if self.ema_decay is None else self.embedding.detach()
        tau = self.log_tau.exp().clamp(1e-3, 10.0)
        logits = F.normalize(zf, dim=-1) @ F.normalize(E, dim=-1).t()
        probs = F.softmax(logits / tau, dim=-1)                # (BG, K) 余弦度量
        # 硬最近码字 + STE: 前向=z_nearest (随输入变), 梯度直通 encoder
        nearest = probs.argmax(-1)                          # (BG,)
        e_hard = E[nearest]
        z_q = (zf + (e_hard - zf).detach()).view(B, G, D)
        commit = F.mse_loss(zf, e_hard.detach())
        marginal = probs.mean(0)
        entropy = -(marginal * (marginal + 1e-10).log()).sum()
        vq_loss = self.beta * commit + self.gamma * (math.log(self.K) - entropy)

        stats = self.monitor(probs, E)
        if self.training:
            with torch.no_grad():
                self.usage_ema.mul_(0.99).add_(marginal.detach(), alpha=0.01)
                if self.ema_decay is not None:
                    self._ema_update(probs, zf)
        return z_q, {"vq": vq_loss, "commit": commit.detach(),
                     "entropy": entropy.detach(), **stats}

    @torch.no_grad()
    def monitor(self, probs, E):
        # H(C), E[H(C|x)], MI, hard usage, embed var, effective rank
        marginal = probs.mean(0)
        H_C = -(marginal * (marginal + 1e-12).log()).sum()
        pxc = probs / (probs.sum(-1, keepdim=True) + 1e-12)
        H_C_x = (-(pxc * (pxc + 1e-12).log()).sum(-1)).mean()
        hard = probs.argmax(-1)
        hard_cnt = torch.bincount(hard, minlength=self.K).float()
        hard_usage = (hard_cnt > 0).float().mean()
        sv = torch.linalg.svdvals(E - E.mean(0))
        eff_rank = (sv.sum() ** 2 / (sv ** 2).sum().clamp_min(1e-12))
        return dict(
            H_C=H_C.detach(), H_C_x=H_C_x.detach(), MI=(H_C - H_C_x).detach(),
            hard_usage=hard_usage.detach(),
            ppl=torch.exp(H_C).detach(),
            embed_var=E.var(dim=0).mean().detach(),
            eff_rank=eff_rank.detach(),
        )

    @torch.no_grad()
    def _ema_update(self, probs, zf):
        cn = probs.sum(0)
        es = probs.t() @ zf
        if dist.is_available() and dist.is_initialized():
            cn = cn.clone(); es = es.clone()
            dist.all_reduce(cn, op=dist.ReduceOp.SUM)
            dist.all_reduce(es, op=dist.ReduceOp.SUM)
        self._cluster_n.mul_(self.ema_decay).add_(cn, alpha=1 - self.ema_decay)
        self._embed_sum.mul_(self.ema_decay).add_(es, alpha=1 - self.ema_decay)
        n = self._cluster_n.sum()
        ns = (self._cluster_n + 1e-5) / (n + self.K * 1e-5) * n
        self.embedding.data.copy_(self._embed_sum / ns.unsqueeze(1))

    @torch.no_grad()
    def restart_dead(self, z_flat, threshold):
        """死码重启: 同步 usage → rank0 选替换 → broadcast。"""
        if dist.is_available() and dist.is_initialized():
            usage = self.usage_ema.clone()
            dist.all_reduce(usage, op=dist.ReduceOp.SUM)
            usage /= dist.get_world_size()
        else:
            usage = self.usage_ema
        dead = usage < threshold
        n = int(dead.sum())
        if n == 0 or n >= self.K:
            return 0
        if not dist.is_initialized() or dist.get_rank() == 0:
            idx = torch.randint(0, z_flat.shape[0], (n,), device=z_flat.device)
            self.embedding.data[dead] = \
                z_flat[idx] + 0.01 * torch.randn(n, self.D, device=z_flat.device)
            if self.ema_decay is not None:
                self._cluster_n.data[dead] = n / self.K
                self._embed_sum.data[dead] = self.embedding.data[dead] * (n / self.K)
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(self.embedding.data, src=0)
        self.usage_ema[dead] = 1.0 / self.K
        return n


# ------------------------------------------------------------------ model
class GeneTokenizer(nn.Module):
    """(u, s, u-s, u_mask, s_mask) → d_model per-gene token。"""
    def __init__(self, spec: GFG3Spec, n_feat: int = 5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_feat, spec.gene_mlp_hidden[0]),
                                 nn.GELU(), nn.LayerNorm(spec.gene_mlp_hidden[0]))

    def forward(self, feats):
        return self.net(feats)


class StateTower(nn.Module):
    """原版 manifold_encoder: per-gene MLP → set attention → z_s (B,G,D)。"""
    def __init__(self, spec: GFG3Spec):
        super().__init__()
        self.tokenize = GeneTokenizer(spec)
        hidden = list(spec.gene_mlp_hidden[1:]) + [spec.d_model]
        self.mlp = gene_mlp(spec.gene_mlp_hidden[0], hidden[:-1], spec.d_model) \
            if len(hidden) > 1 else nn.Identity()
        self.core = SetCore(spec)

    def forward(self, feats):
        h = self.tokenize(feats)
        h = self.mlp(h)
        return self.core(h)


class VelocityTower(nn.Module):
    """原版 velocity_encoder: 测量+state 证据 → z_v (B,G,D)。"""
    def __init__(self, spec: GFG3Spec):
        super().__init__()
        self.detach_state = spec.detach_state_for_velocity
        self.tokenize = GeneTokenizer(spec, n_feat=5 + spec.d_model)
        hidden = list(spec.gene_mlp_hidden[1:]) + [spec.d_model]
        self.mlp = gene_mlp(spec.gene_mlp_hidden[0], hidden[:-1], spec.d_model) \
            if len(hidden) > 1 else nn.Identity()
        self.core = SetCore(spec)

    def forward(self, feats, state):
        if self.detach_state:
            state = state.detach()
        h = self.tokenize(torch.cat([feats, state], -1))
        h = self.mlp(h)
        return self.core(h)


class SharedGeneDecoder(nn.Module):
    """原版 BaseDecoder: 共享逐基因 MLP, 输出 (u_hat, s_hat)。"""
    def __init__(self, spec: GFG3Spec):
        super().__init__()
        dims = [spec.d_model, *spec.decoder_hidden]
        L, w = [], dims[0]
        for h in dims[1:]:
            L += [nn.Linear(w, h), nn.GELU()]; w = h
        L += [nn.Linear(w, 2)]           # 输出层无激活
        self.net = nn.Sequential(*L)

    def forward(self, z):
        return self.net(z)               # (B,G,2) = (u_hat, s_hat)


class GFG3(nn.Module):
    """gene tokens (u,s) → 双塔 → 双 SoftVQ → 共享 decoder → JVP 速度。"""
    def __init__(self, num_genes: int, spec: GFG3Spec, use_vq: bool = True):
        super().__init__()
        spec.validate()
        self.G, self.spec, self.use_vq = int(num_genes), spec, use_vq
        self.state_tower = StateTower(spec)
        self.vel_tower = VelocityTower(spec)
        if use_vq:
            kw = dict(ema_decay=spec.vq_ema_decay)
            self.code_s = SoftVQ(spec.K, spec.d_model, spec.vq_beta,
                                 spec.vq_gamma, spec.vq_tau, **kw)
            self.code_v = SoftVQ(spec.K, spec.d_model, spec.vq_beta,
                                 spec.vq_gamma, spec.vq_tau, **kw)
        self.decoder = SharedGeneDecoder(spec)
        self.restart_threshold = 0.1 / max(self.spec.K, 1)  # 随 K 缩放 (审计 BUG-5)

    # -- 速度定义: 训练与推理唯一 (不变量 7/9)
    def decode_state(self, z_s):
        return self.decoder(z_s)                     # (B,G,2) = (û,ŝ)

    def jvp_velocity(self, z_s, z_v):
        _, v = torch.autograd.functional.jvp(
            self.decoder, (z_s,), (z_v,), create_graph=torch.is_grad_enabled())
        return v                                     # (B,G,2) = (v_u, v_s)

    def _quantize(self, z, codebook):
        if not self.use_vq:
            return z, {"vq": torch.zeros((), device=z.device)}
        zq, d = codebook(z)
        return zq, d

    def forward(self, u, s, compute_velocity: bool = True):
        """u, s: (B, G) 标准化表达; 缺失基因以 NaN/负值传入并自动 mask。"""
        u_m = (u < -50).float(); s_m = (s < -50).float()   # 哨兵 -100 (合法负 z 保留)
        u_c = torch.where(u < -50, torch.zeros_like(u), u)
        s_c = torch.where(s < -50, torch.zeros_like(s), s)
        feats_s = torch.stack((u_c, s_c, u_c - s_c, u_m, s_m), -1)
        z_s_raw = self.state_tower(feats_s)
        z_s, d_s = self._quantize(z_s_raw, getattr(self, "code_s", None))
        rec = self.decode_state(z_s)                 # (B,G,2)
        if compute_velocity:
            z_v_raw = self.vel_tower(feats_s, z_s_raw)
            z_v, d_v = self._quantize(z_v_raw, getattr(self, "code_v", None))
            v_x = self.jvp_velocity(z_s, z_v)        # (B,G,2) 训练=推理同一算子
        else:
            z_v_raw = None
            z_v = torch.zeros_like(z_s)
            d_v = {"vq": z_s.new_zeros(())}
            v_x = torch.zeros_like(rec)
        return dict(rec=rec, v_x=v_x, z_s=z_s, z_v=z_v, d_s=d_s, d_v=d_v,
                    z_s_raw=z_s_raw, z_v_raw=z_v_raw)

    # -- staged training (不变量: 分阶段)
    def set_stage(self, stage: str):
        if stage not in {"state", "velocity", "joint"}:
            raise ValueError(stage)
        st = stage in {"state", "joint"}
        ve = stage in {"velocity", "joint"}
        self.state_tower.requires_grad_(st)
        self.decoder.requires_grad_(st)
        self.vel_tower.requires_grad_(ve)
        self._stage = stage
        if self.use_vq and self.spec.vq_ema_decay is None:
            # 梯度式码本: state 码本随 Stage A, velocity 码本随 Stage B
            self.code_s.requires_grad_(st)
            self.code_v.requires_grad_(ve)
        if self.use_vq:
            # ``model.train()`` is called at the start of every epoch and
            # would otherwise reactivate usage/EMA updates in the frozen
            # stream.  Keep codebook mode aligned with the stage as well.
            self.code_s.train(st)
            self.code_v.train(ve)
        # EMA 码本 requires_grad 恒为 False (由 EMA 更新)
        self.state_tower.train(st); self.decoder.train(st)
        self.vel_tower.train(ve)
        return [p for p in self.parameters() if p.requires_grad]

    def restart_dead_codes(self, z_s_raw, z_v_raw=None):
        if not self.use_vq or not self.training:
            return
        if self.code_s.training:
            self.code_s.restart_dead(z_s_raw.reshape(-1, self.spec.d_model),
                                     self.restart_threshold)
        if self.code_v.training and z_v_raw is not None:
            self.code_v.restart_dead(z_v_raw.reshape(-1, self.spec.d_model),
                                     self.restart_threshold)


# ------------------------------------------------------------------ utils
def count_params(model: nn.Module) -> dict:
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters()
                             if p.requires_grad)}


def build(num_genes: int, config: str = "medium", use_vq: bool = True,
          overrides: dict | None = None) -> GFG3:
    import dataclasses
    spec = dataclasses.replace(CONFIGS[config], **(overrides or {})).validate()
    return GFG3(num_genes, spec, use_vq=use_vq)
