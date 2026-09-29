#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GFG-v3 pilot 评测: held-out donor 上的重构/动力学/几何/稳定性指标。

全部主结果来自 test donor (未参与训练/词表拟合)。
scVelo 速度仅作 teacher agreement (明确非真值)。
"""
import argparse, glob, json, sys
import numpy as np
import torch
import torch.nn.functional as F
import h5py

sys.path.insert(0, "/home/yuchang/GFG_velocity_foundation_mousebrain_v1")
sys.path.insert(0, "/home/yuchang/data_download")
from gfg3_model import build
from gfg3_data import _read_layers, file_vocabulary

AP = argparse.ArgumentParser()
AP.add_argument("--ckpt", default="/data/dataset/gfg3_pilot/final.pt")
AP.add_argument("--max-cells-per-file", type=int, default=20000)
AP.add_argument("--device", default=None)
args = AP.parse_args()

DEV = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
ck_args = ck["args"]
gene_names = ck.get("gene_names")
if gene_names is None:
    gene_names = ck["genes"]   # 旧 ckpt 兼容(索引→需词表)
gene_names = np.asarray(gene_names)
u_mu, u_sd, s_mu, s_sd = (ck["norm_stats"][k] for k in ("u_mu", "u_sd", "s_mu", "s_sd"))
model = build(len(gene_names), ck_args["config"], use_vq=not ck_args.get("no_vq", False))
model.load_state_dict(ck["model"]); model = model.to(DEV).eval()
G = len(gene_names)
print(f"ckpt={args.ckpt} config={ck_args['config']} G={G}")

test_files = ck["splits_manifest"]["files"]["test"]
print(f"test donors: {len(test_files)}")

import scanpy as sc, anndata as ad, scipy.sparse as sp
import scvelo as scv

def cos(a, b, eps=1e-9):
    return float(((a*b).sum(-1) / (np.linalg.norm(a, axis=-1) *
                 np.linalg.norm(b, axis=-1) + eps)).mean())

rows = []
for ti, p in enumerate(test_files):
    v = file_vocabulary(p)
    vp = {g: i for i, g in enumerate(v)}
    m = np.array([vp.get(g, -1) for g in gene_names], dtype=np.int64)
    present = m >= 0
    with h5py.File(p, "r") as f:
        n = f["layers/spliced/indptr"].shape[0] - 1
    r1 = min(n, args.max_cells_per_file)
    if r1 < 2:
        print(f"  [{ti}] 跳过过小文件: cells={r1}")
        continue
    with h5py.File(p, "r") as f:
        u, s = _read_layers(f, 0, r1, len(v))
    up = np.zeros((r1, G), np.float32); sp_ = np.zeros((r1, G), np.float32)
    up[:, present] = u[:, m[present]]; sp_[:, present] = s[:, m[present]]
    uz = np.clip((up - u_mu) / u_sd, -10, 10).astype(np.float32); uz[:, ~present] = -100.0
    sz = np.clip((sp_ - s_mu) / s_sd, -10, 10).astype(np.float32); sz[:, ~present] = -100.0
    U = torch.from_numpy(uz).to(DEV); S = torch.from_numpy(sz).to(DEV)

    # ---- 模型场 (分块) ----
    Vm, rec_s, rec_u = [], [], []
    with torch.no_grad():
      for i in range(0, r1, 512):
        out = model(U[i:i+512], S[i:i+512])
        Vm.append(out["v_x"].cpu().numpy()); rec_s.append(out["rec"][:,:,1].cpu().numpy())
        rec_u.append(out["rec"][:,:,0].cpu().numpy())
      # no_grad 块结束
    Vm = np.concatenate(Vm); RS = np.concatenate(rec_s); RU = np.concatenate(rec_u)
    Vu = Vm[:, :, 0]; Vs = Vm[:, :, 1]

    # teacher: scVelo (per-file)
    a = ad.AnnData(X=sp_.copy(),
                   layers={"spliced": sp_.copy(), "unspliced": up.copy()})
    ok = np.ones(r1, dtype=bool)
    try:
        # 手动归一化 (scanpy normalize_total 不支持 layers=; 与 gen_pairs 同款)
        for ly, src in (("spliced", sp_), ("unspliced", up)):
            lib = a.layers[ly].sum(1, keepdims=True)
            lib[lib == 0] = 1.0
            a.layers[ly] = np.log1p(src / lib * 1e4).astype(np.float32)
        a.X = a.layers["spliced"].copy()
        n_pcs = min(30, r1 - 2, G)
        n_neighbors = min(20, r1 - 1)
        if n_pcs < 1 or n_neighbors < 1:
            raise ValueError("too few cells/genes for teacher kNN")
        sc.pp.pca(a, n_comps=n_pcs); sc.pp.neighbors(a, n_neighbors=n_neighbors)
        scv.pp.moments(a, n_pcs=n_pcs, n_neighbors=n_neighbors)
        scv.tl.velocity(a, mode="stochastic")
        VT = a.layers["velocity"]
        VT = VT.toarray() if sp.issparse(VT) else np.asarray(VT)
        VT = np.nan_to_num(VT)
        ok = np.linalg.norm(VT, axis=1) > 1e-8
        teacher_cos_s = cos(Vs[ok], VT[ok])
        pca = a.obsm["X_pca"].astype(np.float32)
    except Exception as e:
        print(f"  [{ti}] teacher 失败: {e}"); teacher_cos_s = np.nan; teacher_cos_full = np.nan
        VT = np.zeros((r1, G), np.float32); ok = np.ones(r1, bool)
        pca = sp_ - sp_.mean(0); pca = pca[:, :min(30, G)]

    # ---- 指标 ----
    msk_s = (sz > -50).astype(np.float32); msk_u = (uz > -50).astype(np.float32)
    mse_s = float((((RS - sz) ** 2) * msk_s).sum() / msk_s.sum())
    mse_u = float((((RU - uz) ** 2) * msk_u).sum() / msk_u.sum())
    mag = np.linalg.norm(np.concatenate([Vu, Vs], 1), axis=1)
    zero_frac = float((mag < 0.05 * (np.linalg.norm(VT, axis=1).mean() if 'VT' in dir() else 1)).mean())
    # VeloCoh: 邻域速度一致 (kNN20 in PCA)
    from sklearn.neighbors import NearestNeighbors
    ok_idx = np.arange(r1) if 'ok' not in dir() else np.where(ok)[0]
    knn_k = min(21, r1)
    if knn_k < 2:
        print(f"  [{ti}] 跳过邻域指标: cells={r1}")
        continue
    nbr = NearestNeighbors(n_neighbors=knn_k).fit(pca).kneighbors(pca)[1][:, 1:]
    def velcoh(V):
        c = [(V[i] * V[nbr[i]]).sum(1) /
             (np.linalg.norm(V[i]) * np.linalg.norm(V[nbr[i]], axis=1) + 1e-9) for i in ok_idx]
        return float(np.concatenate(c).mean())
    Vmodel_full = np.concatenate([Vu, Vs], 1)
    vc_model, vc_teacher = velcoh(Vmodel_full), velcoh(VT)   # 各自空间内部一致性
    knn_v = VT[nbr].mean(1)                                   # s 空间 (3000)
    agree_knn = cos(Vs[ok], knn_v[ok])                        # s 分量对 s 共识
    # rollout 10 步: 基因空间欧拉, 漂移=到本文件数据云的 PCA 距离
    from sklearn.decomposition import PCA as PCA_
    pp = PCA_(n_components=min(30, r1 - 1, 2 * G), random_state=0).fit(np.concatenate([uz, sz], 1))
    Z = pp.transform(np.concatenate([uz, sz], 1))
    x = Z.copy(); drift = []
    for st in range(10):
        vd = pp.transform(np.concatenate([Vu, Vs], 1))[: len(x)]
        x = x + vd[: len(x)]
        from sklearn.neighbors import NearestNeighbors as NN
        drift.append(float(NN(n_neighbors=min(2, len(Z))).fit(Z).kneighbors(x)[0].mean()))
    rows.append(dict(file=p.split("/")[-1][:14], cells=r1,
                     mse_s=mse_s, mse_u=mse_u, kinetic=float(np.nan),
                     v_mag_mean=float(mag.mean()), v_mag_std=float(mag.std()),
                     zero_frac=zero_frac,
                     teacher_cos_s=teacher_cos_s,
                     velcoh_model=vc_model, velcoh_teacher=vc_teacher,
                     agree_knn=agree_knn, drift10=drift[-1], drift1=drift[0]))
    print(f"[{ti+1}/{len(test_files)}] {rows[-1]}", flush=True)

if not rows:
    raise RuntimeError("no usable test donor/file was evaluated")
ok_rows = [r for r in rows if not np.isnan(r["teacher_cos_s"])]
agg = {k: float(np.nanmean([r[k] for r in ok_rows]))
       for k in rows[0] if k not in ("file", "cells", "drift10", "drift1")}
agg["drift10"] = float(np.nanmean([r["drift10"] for r in rows]))
agg["drift1"] = float(np.nanmean([r["drift1"] for r in rows]))
print("\n======== PILOT 泛化汇总 (held-out donors) ========")
for k, v in agg.items(): print(f"{k:20s} {v:.4f}")
json.dump({"per_file": rows, "agg": agg},
          open(args.ckpt.replace("final.pt", "eval_report.json"), "w"), indent=1)
print(f"-> {args.ckpt.replace('final.pt', 'eval_report.json')}")
