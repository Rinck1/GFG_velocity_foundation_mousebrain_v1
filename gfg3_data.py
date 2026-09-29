#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GFG-v3 数据管线: donor 级划分 + split 后拟合词表/统计 + 流式读取。

数据协议 (规格书三):
  1. 按 source_sample (donor/sample) 划分 train/val/test, 无细胞泄漏。
  2. 划分之后才拟合: 基因词表/HVG、mean/std。
  3. 统一词表 + 缺失基因 mask (缺失以负值传入模型, 模型内转 mask)。
  4. 保存 gene names/顺序/统计/版本 hash。
  5. 流式 (h5py 行块), 不整载 12M 细胞。
  6. 3000 基因 pilot, 可扩 8000/全词表。
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import dataclass

import h5py
import numpy as np
import torch


def file_vocabulary(path: str) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.array([x.decode() if isinstance(x, bytes) else x
                         for x in f["var/_index"][:]])


def filter_readable(paths: list[str], verbose: bool = True) -> list[str]:
    """过滤无法打开的 h5ad (截断/损坏), 返回可用文件列表。"""
    good, bad = [], []
    for p in paths:
        try:
            with h5py.File(p, "r") as f:
                _ = f["var/_index"][:1]
                for layer in ("spliced", "unspliced"):
                    group = f[f"layers/{layer}"]
                    if group["indptr"].ndim != 1 or group["indices"].ndim != 1:
                        raise ValueError(f"{layer} is not CSR")
                    if group["indptr"].shape[0] < 1:
                        raise ValueError(f"{layer} has no indptr")
            good.append(p)
        except Exception as e:
            bad.append((p, type(e).__name__))
    if verbose and bad:
        print(f"[filter_readable] 剔除 {len(bad)} 个坏文件: {[b[0].split('/')[-1] for b in bad[:5]]}")
    return good


def audit_vocab(paths: list[str]) -> dict:
    """词表审计: 统计词表种类 (不再硬性要求逐字节相同, union 词表由 fit 阶段处理)。"""
    seen = {}
    versions = {}
    for p in paths:
        v = file_vocabulary(p)
        h = hashlib.md5(v.tobytes()).hexdigest()[:12]
        seen.setdefault(h, v)
        versions[p] = (h, v)
    if not seen:
        return {"n_files": 0, "vocab_size": 0, "mismatched": [],
                "n_variants": 0, "ref_vocab": np.array([], dtype=str)}
    ref = max(seen.values(), key=len)   # 最大词表作为参考
    mismatched = [p for p, (_, v) in versions.items()
                  if len(v) != len(ref) or not np.array_equal(v, ref)]
    return {"n_files": len(paths), "vocab_size": len(ref),
            "mismatched": mismatched,
            "n_variants": len(seen), "ref_vocab": ref}


def donor_split(paths: list[str], ratios=(0.9, 0.05, 0.05), seed: int = 1234):
    """文件级 (=donor/sample 级) 划分, 同一 source_sample 的细胞不跨集。"""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    n = len(paths)
    n_tr = int(n * ratios[0]); n_va = int(n * ratios[1])
    return ([paths[i] for i in order[:n_tr]],
            [paths[i] for i in order[n_tr:n_tr + n_va]],
            [paths[i] for i in order[n_tr + n_va:]])


def fit_vocab_stats(train_paths: list[str], n_genes: int, block: int = 8192,
                    cache: str | None = None, force: bool = False):
    """在训练集上拟合基因词表 (top-N 高变异) 与逐基因 z-score 统计 (u, s 各自)。

    统计口径: 每层 CP10K + log1p 后的逐基因均值/方差。
    """
    expected_hash = hashlib.md5("".join(train_paths).encode()).hexdigest()[:16]
    if cache and os.path.exists(cache) and not force:
        z = np.load(cache, allow_pickle=True)
        valid_cache = (
            "gene_names" in z.files and "schema_version" in z.files
            and int(np.asarray(z["schema_version"]).item()) >= 2
            and str(np.asarray(z["file_hash"]).item()) == expected_hash
            and "n_genes" in z.files
            and int(np.asarray(z["n_genes"]).item()) == len(z["gene_names"])
            and "requested_n_genes" in z.files
            and int(np.asarray(z["requested_n_genes"]).item()) == int(n_genes)
        )
        if valid_cache:
            return (z["gene_names"], z["genes"], z["u_mu"], z["u_sd"],
                    z["s_mu"], z["s_sd"], z["file_hash"])
        # 旧缓存可能没有显式 schema 或使用了错误的列映射，必须重算。
        z.close()
    return _fit(train_paths, n_genes, block, cache, force)


def _fit(train_paths, n_genes, block=8192, cache=None, force=False):
    if not train_paths:
        raise ValueError("cannot fit vocabulary statistics from an empty train split")
    # 词表: 所有训练文件并集。只读取 var/_index，不会把表达矩阵载入内存。
    uni = set()
    for p in train_paths:
        uni |= set(file_vocabulary(p).tolist())
    vocab = np.array(sorted(uni), dtype=str)
    G = len(vocab)
    union_index = {g: i for i, g in enumerate(vocab.tolist())}
    S1 = np.zeros((2, G)); S2 = np.zeros((2, G)); cnt = 0
    for p in train_paths:
        source_vocab = file_vocabulary(p)
        source_to_union = np.array([union_index[g] for g in source_vocab], dtype=np.int64)
        with h5py.File(p, "r") as f:
            n = f["layers/spliced/indptr"].shape[0] - 1
            for r0 in range(0, n, block):
                r1 = min(r0 + block, n)
                su = _read_layers(f, r0, r1, G, column_map=source_to_union)
                for li, X in enumerate(su):
                    S1[li] += X.sum(0); S2[li] += (X ** 2).sum(0)
                cnt += r1 - r0
    mean = S1 / max(cnt, 1); var = np.maximum(S2 / max(cnt, 1) - mean ** 2, 0)
    order = np.argsort(-var[0])
    order = [i for i in order if mean[0][i] > 0][:n_genes]
    genes = np.array(sorted(order))
    fh = hashlib.md5("".join(train_paths).encode()).hexdigest()[:16]
    payload = dict(genes=genes, gene_names=vocab[genes],
                   u_mu=mean[1][genes], u_sd=np.sqrt(np.maximum(var[1][genes], 1e-8)),
                   s_mu=mean[0][genes], s_sd=np.sqrt(np.maximum(var[0][genes], 1e-8)),
                   file_hash=fh, vocab_size=G,
                   n_genes=np.array(len(genes), dtype=np.int64),
                   requested_n_genes=np.array(n_genes, dtype=np.int64),
                   schema_version=np.array(2, dtype=np.int64))
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, **payload)
    return (payload["gene_names"], payload["genes"], payload["u_mu"], payload["u_sd"],
            payload["s_mu"], payload["s_sd"], payload["file_hash"])


def _read_layers(f, r0, r1, G, column_map: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read CSR rows and return ``(u, s)`` in an explicit target column space.

    ``column_map`` maps a source vocabulary column to a target column.  Keeping
    this mapping at the CSR boundary prevents a reordered or partial h5ad
    vocabulary from silently being interpreted as the first ``G`` target genes.
    """
    if r1 < r0:
        raise ValueError(f"invalid row range [{r0}, {r1})")
    if column_map is None:
        column_map = np.arange(G, dtype=np.int64)
    else:
        column_map = np.asarray(column_map, dtype=np.int64)
    outs = []
    for lyr in (f["layers/spliced"], f["layers/unspliced"]):
        ip = lyr["indptr"][r0:r1 + 1]
        d = lyr["data"][ip[0]:ip[-1]]
        ci = lyr["indices"][ip[0]:ip[-1]]
        B = r1 - r0
        rowid = np.repeat(np.arange(B), np.diff(ip)).astype(np.int64)
        if ci.size:
            target_ci = column_map[np.asarray(ci, dtype=np.int64)]
            keep = target_ci >= 0
            flat = rowid[keep] * G + target_ci[keep]
            weights = np.asarray(d, dtype=np.float64)[keep]
        else:
            flat = np.empty(0, dtype=np.int64)
            weights = np.empty(0, dtype=np.float64)
        X = np.bincount(flat, weights=weights,
                        minlength=B * G).reshape(B, G).astype(np.float32)
        lib = X.sum(1, keepdims=True); lib[lib == 0] = 1.0
        outs.append(np.log1p(X / lib * 1e4))
    # _read_layers 按调用参数先 spliced 后 unspliced; 调用方语义是 (u, s) → 单点对调
    return outs[1], outs[0]


@dataclass
class Split:
    name: str
    files: list[str]
    genes: np.ndarray       # panel 基因在词表中的列索引 (严格同序)
    u_mu: np.ndarray; u_sd: np.ndarray
    s_mu: np.ndarray; s_sd: np.ndarray
    hash: str


class BlockStream:
    """块级流式: 每步产出一个文件的连续行块 (高效 h5py 读), 文件/块顺序受控洗牌。

    返回 dict(u=(B,G), s=(B,G)) 已 z-score; 缺失基因 = -1 (模型内转 mask)。
    """
    def __init__(self, split: Split, block: int = 1024, shuffle: bool = True,
                 seed: int = 0, exclude_tail_frac: float = 0.0,
                 rank: int = 0, world_size: int = 1):
        self.split = split; self.block = block; self.shuffle = shuffle
        self.seed = seed; self.exclude_tail_frac = exclude_tail_frac
        self.rank = int(rank); self.world_size = int(world_size)
        if self.rank < 0 or self.world_size <= 0 or self.rank >= self.world_size:
            raise ValueError(f"invalid shard rank/world_size: {self.rank}/{self.world_size}")
        self.vocabs = [file_vocabulary(p) for p in split.files]
        self.vocab = self.vocabs[0]   # compatibility; maps are per-file below
        self.epoch = 0
        self._build_maps()

    def _blocks(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        out = []
        for fi, p in enumerate(self.split.files):
            with h5py.File(p, "r") as f:
                n = f["layers/spliced/indptr"].shape[0] - 1
            n -= int(n * self.exclude_tail_frac)
            for r0 in range(0, n, self.block):
                out.append((fi, r0, min(r0 + self.block, n)))
        if self.shuffle:
            rng.shuffle(out)
        # Shard complete blocks, never individual cells, so every rank sees a
        # disjoint deterministic stream while preserving CSR read locality.
        return out[self.rank::self.world_size]

    def __iter__(self):
        for fi, r0, r1 in self._blocks():
            with h5py.File(self.split.files[fi], "r") as f:
                u, s = _read_layers(f, r0, r1, len(self.vocabs[fi]))
            m = self.maps[fi]
            present = m >= 0
            u_p = np.zeros((r1 - r0, len(m)), np.float32)
            s_p = np.zeros((r1 - r0, len(m)), np.float32)
            u_p[:, present] = u[:, m[present]]
            s_p[:, present] = s[:, m[present]]
            # z-score 后 clip ±10 (u 层稀疏, sd→0 会产生巨大离群值); 缺失哨兵 -100
            uz = np.clip((u_p - self.split.u_mu) / self.split.u_sd, -10, 10).astype(np.float32)
            sz = np.clip((s_p - self.split.s_mu) / self.split.s_sd, -10, 10).astype(np.float32)
            uz[:, ~present] = -100.0; sz[:, ~present] = -100.0
            yield dict(u=torch.from_numpy(uz), s=torch.from_numpy(sz))
        self.epoch += 1

    def __len__(self):
        return len(self._blocks())

    def _build_maps(self):
        self.maps = []
        for p in self.split.files:
            v = file_vocabulary(p)
            vp = {g: i for i, g in enumerate(v)}
            self.maps.append(np.array([vp.get(g, -1) for g in self.split.genes],
                                      dtype=np.int64))


def save_manifest(path: str, splits: dict, vocab_size: int, genes,
                  gene_names, stats: dict, extra: dict | None = None):
    payload = {
        "data_version_hash": hashlib.md5(
            json.dumps([s.files for s in splits.values()], sort_keys=True).encode()
        ).hexdigest()[:16],
        "vocab_size": vocab_size,
        "genes": genes.tolist(),
        "gene_names": gene_names.tolist(),
        "splits": {k: {"files": v.files, "n_files": len(v.files)} for k, v in splits.items()},
        "stats": stats,
        "extra": extra or {},
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    return payload
