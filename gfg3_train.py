#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GFG-v3 训练入口: 分阶段 (A state / B velocity / C joint) + DDP + strict resume。

规格书五:
  Stage A: 掩码 s/u 重构 (不动 scVelo 速度)
  Stage B: 冻结/低 lr 状态流, 训练 velocity 流 (kinetic + graph smooth/align)
           scVelo 仅作可选 weak teacher, 单独记 teacher_agreement
  Stage C: 渐进解冻, loss 权重 ramp
  各项损失梯度尺度自动归一化 (不沿用固定 20/300/300)
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import h5py
import torch
import torch.nn.functional as F
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gfg3_model import build, count_params, GFG3Spec, CONFIGS, SoftVQ
from gfg3_data import (audit_vocab, donor_split, fit_vocab_stats, Split,
                       BlockStream, save_manifest, filter_readable)
from gfg3_moments import MomentsStream, fit_moments_stats
from model.Decoder import ode_shared_gene_residual_loss  # noqa: E402
from gfg3_data import file_vocabulary, _read_layers


# ---------------- 梯度尺度均衡器 ----------------
class GradNormBalancer:
    """周期测量各项损失对共享参数的梯度范数, 权重 = target / (ema_norm + eps)。

    替代规格书五中"不沿用固定 20/300/300"的要求。
    """
    def __init__(self, terms, target_norm=1.0, ema=0.95, measure_every=100):
        self.terms = list(terms)
        self.target = target_norm
        self.ema_f = ema
        self.measure_every = measure_every
        self.norms = {t: 1.0 for t in terms}
        self.weights = {t: 1.0 for t in terms}

    def measure(self, model, terms: dict):
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return
        for name, loss in terms.items():
            if not torch.is_tensor(loss) or not loss.requires_grad:
                self.norms[name] = self.norms[name]
                continue
            gs = torch.autograd.grad(loss, params, retain_graph=True,
                                     allow_unused=True)
            sq = sum((g ** 2).sum() for g in gs if g is not None)
            if not torch.is_tensor(sq):
                sq = loss.new_zeros(())
            self.norms[name] = (self.ema_f * self.norms[name]
                                + (1 - self.ema_f) * float(sq.sqrt()))
        if dist.is_initialized():
            device = params[0].device
            t_ = torch.tensor([self.norms[t] for t in self.terms], device=device)
            dist.all_reduce(t_, op=dist.ReduceOp.AVG)
            for i, t in enumerate(self.terms): self.norms[t] = float(t_[i])
        for t in self.terms:
            self.weights[t] = min(self.target / (self.norms[t] + 1e-8), 1e4)

    def maybe_measure(self, step, model, terms):
        if step % self.measure_every == 0:
            self.measure(model, terms)

    def scaled(self, name, loss):
        return self.weights[name] * loss


# ---------------- DDP ----------------
def ddp_init():
    if "RANK" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("WORLD_SIZE>1 requires CUDA")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
        return int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    return 0, 0, 1


def is_main():
    return dist.get_rank() == 0 if dist.is_initialized() else True


def log(*a):
    if is_main():
        print(*a, flush=True)


# ---------------- checkpoint (规格书七) ----------------
def save_ckpt(path, model, opt, args, genes, gene_names, splits_meta, stats,
              rng_state=None, sched=None, stage=None, epoch=-1, gstep=0):
    raw = model.module if hasattr(model, "module") else model
    if rng_state is None:
        rng_state = {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "numpy": np.random.get_state(),
        }
    payload = {
        "schema_version": 2,
        "model": raw.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict() if sched is not None else None,
        "stage": stage,
        "epoch": int(epoch),
        "gstep": int(gstep),
        "args": vars(args),
        "config": {**vars(args), "spec": asdict_safe(raw.spec)},
        "genes": genes, "gene_names": gene_names,
        "splits_manifest": splits_meta,
        "norm_stats": {k: v for k, v in stats.items()},
        "codebook_ema_state": {
            name: dict(
                cluster_n=getattr(mod, "_cluster_n", None),
                embed_sum=getattr(mod, "_embed_sum", None),
                usage_ema=mod.usage_ema,
            )
            for name, mod in raw.named_modules() if isinstance(mod, SoftVQ)
        },
        "rng": rng_state,
        "code_hash": code_hash(),
    }
    torch.save(payload, path)


def asdict_safe(spec):
    from dataclasses import asdict
    return asdict(spec)


def code_hash():
    import gfg3_model, gfg3_data
    h = hashlib.md5()
    for f in (gfg3_model.__file__, gfg3_data.__file__, __file__):
        h.update(open(f, "rb").read())
    return h.hexdigest()[:16]


# ---------------- 损失 ----------------
def stage_a_losses(out, u, s):
    # 掩码重构: 缺失位置 (-1 输入) 不计损失
    mask_u = (u > -50).float(); mask_s = (s > -50).float()   # 缺失哨兵=-100
    l_s = ((out["rec"][:, :, 1] - s) ** 2 * mask_s).sum() / mask_s.sum().clamp_min(1)
    l_u = ((out["rec"][:, :, 0] - u) ** 2 * mask_u).sum() / mask_u.sum().clamp_min(1)
    return dict(recon_s=l_s, recon_u=l_u,
                vq_s=out["d_s"]["vq"], vq_v=out["d_v"]["vq"])


def stage_b_losses(out, u, s, nbr=None, stats=None):
    d = stage_a_losses(out, u, s)
    v_u, v_s = out["v_x"][:, :, 0], out["v_x"][:, :, 1]
    # The model works in per-gene z-scored log-expression.  Convert the state
    # and its JVP back to the same log-expression coordinate before applying
    # the RNA ODE.  A missing-gene sentinel is masked out; clamping z-scores to
    # zero changes the physical problem and was a source of false tiny losses.
    valid = (u > -50) & (s > -50)
    if stats is not None:
        dev = u.device
        u_mu = torch.as_tensor(stats["u_mu"], device=dev, dtype=u.dtype)
        u_sd = torch.as_tensor(stats["u_sd"], device=dev, dtype=u.dtype)
        s_mu = torch.as_tensor(stats["s_mu"], device=dev, dtype=u.dtype)
        s_sd = torch.as_tensor(stats["s_sd"], device=dev, dtype=u.dtype)
        kin_u = u * u_sd + u_mu
        kin_s = s * s_sd + s_mu
        kin_v_u = v_u * u_sd
        kin_v_s = v_s * s_sd
    else:
        kin_u, kin_s, kin_v_u, kin_v_s = u, s, v_u, v_s
    kin_u = torch.where(valid, kin_u, torch.zeros_like(kin_u))
    kin_s = torch.where(valid, kin_s, torch.zeros_like(kin_s))
    kin_v_u = torch.where(valid, kin_v_u, torch.zeros_like(kin_v_u))
    kin_v_s = torch.where(valid, kin_v_s, torch.zeros_like(kin_v_s))
    l_kin, _ = ode_shared_gene_residual_loss(
        kin_u, kin_s, kin_v_u, kin_v_s, mask=valid, nonneg="none"
    )
    d["kinetic"] = l_kin
    if nbr is not None:
        # smooth: 邻居速度方向余弦 (展平 G×2 联合速度向量)
        v = out["v_x"].reshape(out["v_x"].shape[0], -1)   # (B, 2G)
        v_nb = nbr["v_x"].reshape(*nbr["v_x"].shape[:2], -1)  # (B,k,2G)
        cos = F.cosine_similarity(v.unsqueeze(1), v_nb, dim=-1)   # (B,k)
        d["smooth"] = (1 - cos).mean()
        # directed align: v 指向邻居表达位置 (同布局展平)
        # v_x is (B,G,2), hence gene-interleaved.  Graph samples are loaded as
        # (B,k,2,G) for the two expression layers and must be transposed to the
        # same (gene,layer) layout before cosine alignment.
        x_nb = nbr["x"].permute(0, 1, 3, 2).reshape(*nbr["x"].shape[:2], -1)
        x_self = torch.stack((u, s), dim=-1).reshape(u.shape[0], -1)
        cos2 = F.cosine_similarity(v.unsqueeze(1), x_nb - x_self.unsqueeze(1), dim=-1)
        d["align"] = (1 - cos2).mean()
    return d


# ---------------- 图缓存与批采样 (真实 kNN 图, 测试7) ----------------
class GraphCache:
    """Memory-bounded per-file kNN cache.

    A graph is built on a deterministic representative subset of each donor,
    not on the complete dense h5ad matrix.  The cached neighbour ids are local
    to that subset and are mapped back to source rows only when sampled.
    """
    def __init__(self, files, genes, u_mu, u_sd, s_mu, s_sd, k=20, n_pcs=30,
                 moments_dir=None, min_cells=50, max_cells_per_file=4096,
                 seed=0):
        from sklearn.neighbors import NearestNeighbors
        self.k = int(k); self.moments_dir = moments_dir
        self.max_cells_per_file = int(max_cells_per_file)
        self.seed = int(seed)
        if self.max_cells_per_file < 2:
            raise ValueError("max_cells_per_file must be at least 2")
        self.meta = []; self.nbr = []
        self.u_mu, self.u_sd, self.s_mu, self.s_sd = u_mu, u_sd, s_mu, s_sd
        rng = np.random.default_rng(self.seed)
        for p in files:
            if moments_dir:
                sid = os.path.basename(p)[:-5]
                if not os.path.exists(os.path.join(moments_dir, sid + ".s.npy")):
                    continue
            v = file_vocabulary(p)
            vp = {g: i for i, g in enumerate(v)}
            m = np.array([vp.get(g, -1) for g in genes], dtype=np.int64)
            present = m >= 0
            if moments_dir:
                sid = os.path.basename(p)[:-5]
                n = np.load(os.path.join(moments_dir, sid + ".s.npy"),
                            mmap_mode="r").shape[0]
                if n < max(2, min_cells):
                    continue
            else:
                with h5py.File(p, "r") as f:
                    n = f["layers/spliced/indptr"].shape[0] - 1
                if n < max(2, min_cells):
                    continue
            sample_n = min(n, self.max_cells_per_file)
            rows = np.arange(n, dtype=np.int64)
            if sample_n < n:
                rows = np.sort(rng.choice(n, sample_n, replace=False).astype(np.int64))
            u, s = self._load_rows(p, rows, len(v))
            n_graph = len(rows)
            up = np.zeros((n_graph, len(genes)), np.float32)
            sp = np.zeros((n_graph, len(genes)), np.float32)
            up[:, present] = u[:, m[present]]
            sp[:, present] = s[:, m[present]]
            uz = np.clip((up - u_mu) / u_sd, -10, 10); uz[:, ~present] = -100.0
            sz = np.clip((sp - s_mu) / s_sd, -10, 10); sz[:, ~present] = -100.0
            X = np.concatenate([uz, sz], 1).astype(np.float32)
            Xc = sp - sp.mean(0)
            if n_graph > n_pcs + 1:
                from sklearn.decomposition import PCA
                pcs = PCA(n_components=min(n_pcs, n_graph - 1), random_state=0
                          ).fit_transform(Xc)
            else:
                pcs = Xc[:, :n_pcs] if Xc.shape[1] >= n_pcs else np.pad(Xc, ((0,0),(0,n_pcs-Xc.shape[1])))
            kk = min(self.k, n_graph - 1)
            if kk < 1:
                continue
            nn = NearestNeighbors(n_neighbors=kk + 1).fit(pcs).kneighbors(pcs)[1][:, 1:]
            if nn.shape[1] < self.k:   # 邻居数不足则重复补齐
                pad = np.tile(nn[:, :1], (1, self.k - nn.shape[1]))
                nn = np.concatenate([nn, pad], 1)
            # Only local neighbour indices and source row ids are resident.
            self.nbr.append(torch.from_numpy(nn.astype(np.int64)))
            self.meta.append(dict(path=p, n=n, rows=rows, panel_map=m, present=present))
        self.offsets = np.cumsum([0] + [x.shape[0] for x in self.nbr])

    def __len__(self):
        return int(self.offsets[-1]) if len(self.offsets) else 0

    def _load_rows(self, path, rows, source_genes):
        """Read exact rows, grouping contiguous HDF5 ranges to bound I/O."""
        rows = np.asarray(rows, dtype=np.int64)
        if rows.size == 0:
            z = np.empty((0, source_genes), dtype=np.float32)
            return z, z.copy()
        if self.moments_dir:
            sid = os.path.basename(path)[:-5]
            U = np.load(os.path.join(self.moments_dir, sid + ".u.npy"), mmap_mode="r")
            S = np.load(os.path.join(self.moments_dir, sid + ".s.npy"), mmap_mode="r")
            return np.asarray(U[rows]), np.asarray(S[rows])
        order = np.argsort(rows)
        sorted_rows = rows[order]
        u_sorted = np.empty((len(rows), source_genes), dtype=np.float32)
        s_sorted = np.empty_like(u_sorted)
        with h5py.File(path, "r") as f:
            start = 0
            while start < len(sorted_rows):
                end = start + 1
                while end < len(sorted_rows) and sorted_rows[end] == sorted_rows[end - 1] + 1:
                    end += 1
                r0, r1 = int(sorted_rows[start]), int(sorted_rows[end - 1]) + 1
                uu, ss = _read_layers(f, r0, r1, source_genes)
                u_sorted[start:end] = uu
                s_sorted[start:end] = ss
                start = end
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        return u_sorted[inverse], s_sorted[inverse]

    def _cells(self, fi, rows):
        rows = np.asarray(rows, dtype=np.int64)
        source_rows = self.meta[fi]["rows"][rows]
        u, s = self._load_rows(
            self.meta[fi]["path"], source_rows,
            len(file_vocabulary(self.meta[fi]["path"]))
        )
        m = self.meta[fi]["panel_map"]; present = self.meta[fi]["present"]
        up = np.zeros((len(rows), len(self.meta[fi]["panel_map"])), np.float32)
        sp_ = np.zeros_like(up)
        up[:, present] = u[:, m[present]]; sp_[:, present] = s[:, m[present]]
        uz = np.clip((up - self.u_mu) / self.u_sd, -10, 10).astype(np.float32)
        sz = np.clip((sp_ - self.s_mu) / self.s_sd, -10, 10).astype(np.float32)
        uz[:, ~present] = -100.0; sz[:, ~present] = -100.0
        return uz, sz

    def sample(self, n, rng):
        if len(self) == 0:
            raise RuntimeError("graph cache has no file with at least two usable cells")
        idx = rng.integers(0, self.offsets[-1], n)
        fi = np.searchsorted(self.offsets, idx, side="right") - 1
        local = idx - self.offsets[fi]
        A_u, A_s, NB_u, NB_s = [], [], [], []
        for f in sorted(set(fi.tolist())):
            sel = local[fi == f]
            nbrs = self.nbr[f][sel]                       # (b,k)
            rows = np.unique(np.concatenate([sel, nbrs.ravel()]))
            uz, sz = self._cells(f, rows)
            ridx = {r: i for i, r in enumerate(rows)}
            ar = torch.tensor([ridx[int(r)] for r in sel])
            nr = torch.tensor([[ridx[int(c)] for c in row] for row in nbrs])
            a = torch.from_numpy(np.concatenate([uz[ar], sz[ar]], 1))
            nb = torch.from_numpy(np.stack(
                [np.concatenate([uz[nr[b]], sz[nr[b]]], 1) for b in range(len(sel))]))
            A_u.append(a); NB_u.append(nb)
        return torch.cat(A_u), torch.cat(NB_u)


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="/data/dataset/Velocyto")
    ap.add_argument("--n-files", type=int, default=0, help="pilot 文件数 (0=全部)")
    ap.add_argument("--corpus", default=None, help="器官语料清单 json {organ: [ids]}")
    ap.add_argument("--data1-dir", default="/data/dataset/Velocyto")
    ap.add_argument("--steps-per-epoch", type=int, default=2000)
    ap.add_argument("--stats-files", type=int, default=500, help="统计拟合文件数上限")
    ap.add_argument("--graph-files", type=int, default=300, help="图缓存文件数上限")
    ap.add_argument("--graph-cells-per-file", type=int, default=4096,
                    help="每个 donor 用于 kNN 图的代表细胞数上限")
    ap.add_argument("--moments-dir", default=None, help="Ms/Mu moments 缓存目录 (论文忠实输入)")
    ap.add_argument("--min-cells", type=int, default=50)
    ap.add_argument("--n-genes", type=int, default=3000)
    ap.add_argument("--config", default="original", choices=list(CONFIGS))
    ap.add_argument("--no-vq", action="store_true")
    ap.add_argument("--epochs-a", type=int, default=20)
    ap.add_argument("--epochs-b", type=int, default=20)
    ap.add_argument("--epochs-c", type=int, default=0)
    ap.add_argument("--batch", type=int, default=None,
                    help="训练 batch 行数；未指定时使用 --block（旧接口）")
    ap.add_argument("--block", type=int, default=64,
                    help="兼容旧接口的 batch/block 行数")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-state-decay", type=float, default=0.1, help="Stage B 状态流 lr 折扣")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/data/dataset/gfg3_ckpt")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--scvelo-teacher", action="store_true",
                    help="scVelo 仅作 weak teacher, 记录 teacher_agreement")
    args = ap.parse_args()
    effective_batch = int(args.batch if args.batch is not None else args.block)
    if effective_batch <= 0:
        raise ValueError("batch/block must be positive")
    args.effective_batch = effective_batch

    RANK, LOCAL_RANK, WORLD = ddp_init()
    if WORLD > 1 and not torch.cuda.is_available():
        raise RuntimeError("DDP training requires CUDA/NCCL")
    DEV = f"cuda:{LOCAL_RANK}" if torch.cuda.is_available() else "cpu"
    MAIN = RANK == 0
    log = (lambda *a: print(*a, flush=True)) if MAIN else (lambda *a: None)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    import random as _r
    _r.seed(args.seed + RANK)
    organ_of = {}
    if args.moments_dir:
        mf = json.load(open(os.path.join(args.moments_dir, "manifest.json")))
        tr = [f"/data/dataset/Velocyto/{x}.h5ad" for x in mf["train"]]
        va = [f"/data/dataset/Velocyto/{x}.h5ad" for x in mf["val"]]
        te = [f"/data/dataset/Velocyto/{x}.h5ad" for x in mf["test"]]
        tr = [p for p in tr if os.path.exists(p)]
        va = [p for p in va if os.path.exists(p)]
        te = [p for p in te if os.path.exists(p)]
        gene_names, genes, u_mu, u_sd, s_mu, s_sd, fh = fit_moments_stats(
            tr, args.n_genes, args.moments_dir,
            cache=os.path.join(args.out, "vocab_stats_moments.npz"))  # 区分 raw/moments 统计
        log(f"[moments] train={len(tr)} val={len(va)} test={len(te)} panel={len(gene_names)}")
    elif args.corpus:
        corpus = json.load(open(args.corpus))
        all_files, organ_of = [], {}
        for organ, ids in corpus.items():
            for sid in ids:
                p = os.path.join(args.data1_dir, sid + ".h5ad")
                if os.path.exists(p):
                    all_files.append(p); organ_of[p] = organ
        log(f"corpus: {len(all_files)} files, organs=" +
            str({o: sum(1 for x in organ_of.values() if x == o) for o in set(organ_of.values())}))
    else:
        files = sorted(glob.glob(os.path.join(args.data_root, "*.h5ad")))
        all_files = files[:args.n_files] if args.n_files else files
    if not args.moments_dir:
        all_files = filter_readable(all_files)
        def cell_count(path):
            with h5py.File(path, "r") as f:
                return int(f["layers/spliced/indptr"].shape[0] - 1)
        before = len(all_files)
        all_files = [p for p in all_files if cell_count(p) >= args.min_cells]
        log(f"min-cells={args.min_cells}: 保留 {len(all_files)}/{before} 文件")
        if len(all_files) < 3:
            raise ValueError("need at least three readable files for train/val/test")
        if len(all_files) <= 30:
            log("warning: small corpus; donor validation/test estimates will be noisy")
        aud = audit_vocab(all_files[:200])
        log(f"vocab variants (前200文件): {aud['n_variants']} 种 → union 词表")
        groups = [organ_of.get(p, "unknown") for p in all_files] if organ_of else None
        tr, va, te = donor_split(all_files, seed=args.seed, groups=groups)
        log(f"donor split: train={len(tr)} val={len(va)} test={len(te)}")
        cache = os.path.join(args.out, "vocab_stats.npz")
        tr_stats = _r.sample(tr, min(args.stats_files, len(tr)))
        gene_names, genes, u_mu, u_sd, s_mu, s_sd, fh = fit_vocab_stats(
            tr_stats, args.n_genes, cache=cache)
        log(f"panel={len(gene_names)} data_hash={fh}")
    splits_meta = {"train": len(tr), "val": len(va), "test": len(te),
                   "files": {"train": tr, "val": va, "test": te}}
    stats = {"u_mu": u_mu, "u_sd": u_sd, "s_mu": s_mu, "s_sd": s_sd}

    model = build(len(gene_names), args.config, use_vq=not args.no_vq).to(DEV)
    log(f"config={args.config} use_vq={not args.no_vq} params={count_params(model)}")
    if WORLD > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[LOCAL_RANK], find_unused_parameters=True)  # 分阶段冻结必需 (审计 BUG-3)
    raw = model.module if WORLD > 1 else model

    tr_graph = _r.sample(tr, min(args.graph_files, len(tr)))
    graph = GraphCache(tr_graph, gene_names, u_mu, u_sd, s_mu, s_sd,
                       moments_dir=args.moments_dir,
                       min_cells=args.min_cells,
                       max_cells_per_file=args.graph_cells_per_file,
                       seed=args.seed)
    log(f"graph cache: {len(tr_graph)} files, {graph.offsets[-1]} cells")
    # 论文忠实: 码本 5x lr (原版 5e-3 vs 1e-3) + ExponentialLR 0.9
    raw_ = model.module if hasattr(model, "module") else model
    cb_params, other_params = [], []
    for name, prm in raw_.named_parameters():
        (cb_params if "code_" in name and "embedding" in name else other_params).append(prm)
    opt = torch.optim.Adam(
        [{"params": other_params, "lr": args.lr},
         {"params": cb_params, "lr": 5 * args.lr}])
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)
    balancer = GradNormBalancer(["recon_s", "recon_u", "kinetic", "smooth", "align",
                                 "vq_s", "vq_v"])
    start_epoch = 0
    resume_stage_idx = 0
    gstep = 0
    stages_order = ["state", "velocity", "joint"]
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        raw.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if ck.get("scheduler") is not None:
            sched.load_state_dict(ck["scheduler"])
        start_epoch = int(ck.get("epoch", -1)) + 1
        gstep = int(ck.get("gstep", 0))
        resume_stage_idx = stages_order.index(ck.get("stage", "state"))
        log(f"resumed {args.resume}: stage={ck.get('stage')} ep={start_epoch}")

    os.makedirs(args.out, exist_ok=True)
    metrics_f = open(os.path.join(args.out, "metrics.jsonl"), "a") if MAIN else None

    stage_list = [("state", args.epochs_a), ("velocity", args.epochs_b),
                  ("joint", args.epochs_c)]
    completed_stage, completed_epoch = None, -1
    for si_, (stage, epochs) in enumerate(stage_list):
        if epochs <= 0 or si_ < resume_stage_idx:
            continue
        epoch_start = start_epoch if args.resume and si_ == resume_stage_idx else 0
        for ep in range(epoch_start, epochs):
            model.train()
            # model.train() recursively enables every submodule; re-apply the
            # stage freeze/eval policy after it, otherwise dropout and EMA VQ
            # continue updating in frozen streams.
            raw.set_stage(stage)
            streams = [MomentsStream(tr, gene_names, u_mu, u_sd, s_mu, s_sd,
                             args.moments_dir, block=effective_batch, shuffle=True,
                             seed=args.seed + ep * 100, rank=RANK,
                             world_size=WORLD)
                      if args.moments_dir else
                      BlockStream(Split("train", tr, gene_names, u_mu, u_sd, s_mu, s_sd, fh),
                                  block=effective_batch, shuffle=True,
                                  seed=args.seed + ep * 100, rank=RANK,
                                  world_size=WORLD)]
            it = iter(streams[0])
            run = {}
            t0 = time.time()
            n_steps = min(args.steps_per_epoch, max(len(streams[0]), 1))
            for step in range(n_steps):
                batch = next(it)
                u = batch["u"].to(DEV); s = batch["s"].to(DEV)
                out = model(u, s, compute_velocity=(stage != "state"))
                nbr = None
                if stage != "state" and len(graph) > 0:
                    A, NB = graph.sample(len(u), np.random.default_rng(gstep))
                    NB4 = NB.view(len(u), -1, 2, len(gene_names))      # (n,k,2,G)
                    was = raw.training
                    raw.eval()      # 防邻居前向双更新 VQ
                    with torch.no_grad():
                        v_nb = model(NB4[..., 0, :].reshape(-1, len(gene_names)).to(DEV),
                                     NB4[..., 1, :].reshape(-1, len(gene_names)).to(DEV)
                                     )["v_x"]
                    if was:
                        raw.train(); raw.set_stage(stage)
                    nbr = dict(x=NB4.to(DEV),
                               v_x=v_nb.view(len(u), -1, len(gene_names), 2))
                losses = stage_b_losses(out, u, s, nbr=nbr, stats=stats) if stage != "state" \
                    else stage_a_losses(out, u, s)
                balancer.maybe_measure(gstep, raw, losses)
                if stage == "state":
                    loss = (balancer.scaled("recon_s", losses["recon_s"])
                            + balancer.scaled("recon_u", losses["recon_u"])
                            + balancer.scaled("vq_s", losses["vq_s"])
                            + balancer.scaled("vq_v", losses["vq_v"]))
                else:
                    loss = sum(balancer.scaled(k, v) for k, v in losses.items())
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                raw.restart_dead_codes(out["z_s_raw"], out["z_v_raw"])
                gstep += 1
                for k, v in losses.items():
                    run[k] = run.get(k, 0.0) + float(v)
                if MAIN and step % 50 == 0:
                    msg = " ".join(f"{k}={v/(step+1):.4f}" for k, v in run.items())
                    print(f"[{stage} ep{ep} s{step}] {msg} "
                          f"w={ {k: round(v,2) for k,v in balancer.weights.items()} } "
                          f"{time.time()-t0:.0f}s", flush=True)
            sched.step()
            completed_stage, completed_epoch = stage, ep
            if WORLD > 1:
                dist.barrier()
            if MAIN and metrics_f:
                rec = {k: v / max(n_steps, 1) for k, v in run.items()}
                rec.update(stage=stage, epoch=ep, gstep=gstep, steps=n_steps)
                metrics_f.write(json.dumps(rec) + "\n"); metrics_f.flush()
                save_ckpt(
                    os.path.join(args.out, f"{stage}_last.pt"), model, opt, args,
                    genes, gene_names, splits_meta, stats, sched=sched,
                    stage=stage, epoch=ep, gstep=gstep,
                )
            if WORLD > 1:
                dist.barrier()
        # A resumed stage only consumes start_epoch once; later stages start at
        # their own epoch zero.
        start_epoch = 0
    if MAIN:
        save_ckpt(
            os.path.join(args.out, "final.pt"), model, opt, args, genes,
            gene_names, splits_meta, stats, sched=sched,
            stage=completed_stage or "state", epoch=completed_epoch,
            gstep=gstep,
        )
        if metrics_f:
            metrics_f.close()
        log("DONE")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
