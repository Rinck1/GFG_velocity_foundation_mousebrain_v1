#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GFG-v3 moments 预处理: 按论文 C.3.2 还原 Ms/Mu 邻域平滑输入。

流程 (每文件):
  log 归一化 u/s (CP10K+log1p)
  → PCA-30 (子样本拟合, 对 s)
  → kNN-20 → 行归一化邻接 A
  → Ms = A @ S, Mu = A @ U  (scvelo moments 一阶)
  → 存 {u,s}.npy (fp32, mmap 可读)
同时产出分层抽样语料 manifest (每器官 train/val/test donor)。
"""
import argparse, glob, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import h5py

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gfg3_data import file_vocabulary, _read_layers

MIN_CELLS = 50


def build_one(args):
    p, out_dir = args
    sid = os.path.basename(p)[:-5]
    out = os.path.join(out_dir, sid)
    if os.path.exists(out + ".s.npy") and os.path.exists(out + ".u.npy"):
        return sid, "skip"
    try:
        from sklearn.decomposition import PCA
        from sklearn.neighbors import NearestNeighbors
        v = file_vocabulary(p)
        with h5py.File(p, "r") as f:
            n = f["layers/spliced/indptr"].shape[0] - 1
            if n < MIN_CELLS:
                return sid, "tiny"
            u_log, s_log = _read_layers(f, 0, n, len(v))   # (n, G) 已 log 归一化
        # PCA-30 (子样本 ≤40k 拟合, 对 s)
        rng = np.random.default_rng(0)
        sub = rng.choice(n, min(n, 40000), replace=False)
        pca = PCA(n_components=min(30, n - 2, s_log.shape[1]), random_state=0).fit(s_log[sub])
        Z = pca.transform(s_log)
        nn = NearestNeighbors(n_neighbors=min(21, n)).fit(Z).kneighbors(Z)[1][:, 1:]
        # 行归一化邻接平滑
        k = nn.shape[1]
        Ms = s_log[nn].mean(1)
        Mu = u_log[nn].mean(1)
        np.save(out + ".s.npy", Ms.astype(np.float32))
        np.save(out + ".u.npy", Mu.astype(np.float32))
        return sid, "ok"
    except Exception as e:
        return sid, f"err:{type(e).__name__}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="/home/yuchang/data_download/organ_corpus_6.json")
    ap.add_argument("--out", default="/data1/yuchang/moments")
    ap.add_argument("--per-organ", type=int, default=460, help="每器官抽样文件数")
    ap.add_argument("--val-test", type=int, default=23, help="每器官 val/test donor 数")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    corpus = json.load(open(args.corpus))
    rng = np.random.default_rng(args.seed)
    manifest = {"train": [], "val": [], "test": [], "organ": {}}
    for organ, ids in corpus.items():
        ids = list(rng.permutation(sorted(ids)))[:args.per_organ]
        nv = args.val_test
        manifest["test"] += ids[:nv]
        manifest["val"] += ids[nv:2 * nv]
        manifest["train"] += ids[2 * nv:]
        for i in ids:
            manifest["organ"][i] = organ
        print(f"{organ:10s} 抽样 {len(ids)} (train {len(ids)-2*nv} / val {nv} / test {nv})")

    tasks = []
    for split in ("train", "val", "test"):
        for sid in manifest[split]:
            p = f"/data/dataset/Velocyto/{sid}.h5ad"
            if os.path.exists(p):
                tasks.append((p, args.out))
    print(f"moments 待建: {len(tasks)} 文件, workers={args.workers}")
    t0 = time.time(); stat = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (sid, st) in enumerate(ex.map(build_one, tasks, chunksize=4)):
            stat[st.split(":")[0]] = stat.get(st.split(":")[0], 0) + 1
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{len(tasks)} {stat} {time.time()-t0:.0f}s", flush=True)
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"), indent=1)
    print(f"DONE {stat} -> {args.out}/manifest.json")


if __name__ == "__main__":
    main()


# ---------------- 训练侧: moments 流式读取与统计 ----------------
import torch


class MomentsStream:
    """从 moments 缓存 (mmap .npy) 流式产出标准化 (u, s) 块。"""
    def __init__(self, files, gene_names, u_mu, u_sd, s_mu, s_sd, moments_dir,
                 block=64, shuffle=True, seed=0, rank=0, world_size=1):
        self.files = files; self.block = block; self.shuffle = shuffle
        self.seed = seed; self.dir = moments_dir
        self.rank = int(rank); self.world_size = int(world_size)
        if self.rank < 0 or self.world_size <= 0 or self.rank >= self.world_size:
            raise ValueError(f"invalid shard rank/world_size: {self.rank}/{self.world_size}")
        self.gene_names = list(gene_names)
        self.u_mu, self.u_sd, self.s_mu, self.s_sd = u_mu, u_sd, s_mu, s_sd
        self.epoch = 0
        # 词表映射: moments 数组列 = 原文件基因序 (全人类共享词表, 取 vocab[0] 建映射)
        from gfg3_data import file_vocabulary
        v = file_vocabulary(files[0])
        vp = {g: i for i, g in enumerate(v)}
        self.panel_idx = np.array([vp.get(g, -1) for g in self.gene_names], np.int64)
        self.maps = []
        for p in files:
            vv = file_vocabulary(p)
            vvp = {g: i for i, g in enumerate(vv)}
            m = np.array([vvp.get(g, -1) for g in self.gene_names], np.int64)
            self.maps.append(m)

    def _blocks(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        out = []
        for fi, p in enumerate(self.files):
            sid = os.path.basename(p)[:-5]
            sp_ = os.path.join(self.dir, sid + ".s.npy")
            if not os.path.exists(sp_):
                continue
            n = np.load(sp_, mmap_mode="r").shape[0]
            for r0 in range(0, n, self.block):
                out.append((fi, r0, min(r0 + self.block, n)))
        if self.shuffle:
            rng.shuffle(out)
        if out and self.world_size > 1:
            target = ((len(out) + self.world_size - 1) // self.world_size) * self.world_size
            out = out + out[: target - len(out)]
        return out[self.rank::self.world_size]

    def __iter__(self):
        for fi, r0, r1 in self._blocks():
            sid = os.path.basename(self.files[fi])[:-5]
            S = np.load(os.path.join(self.dir, sid + ".s.npy"), mmap_mode="r")[r0:r1]
            U = np.load(os.path.join(self.dir, sid + ".u.npy"), mmap_mode="r")[r0:r1]
            m = self.maps[fi]; present = m >= 0
            up = np.zeros((r1 - r0, len(m)), np.float32)
            sp_ = np.zeros((r1 - r0, len(m)), np.float32)
            up[:, present] = np.asarray(U)[:, m[present]]
            sp_[:, present] = np.asarray(S)[:, m[present]]
            uz = np.clip((up - self.u_mu) / self.u_sd, -10, 10).astype(np.float32)
            sz = np.clip((sp_ - self.s_mu) / self.s_sd, -10, 10).astype(np.float32)
            uz[:, ~present] = -100.0; sz[:, ~present] = -100.0
            yield dict(u=torch.from_numpy(uz), s=torch.from_numpy(sz))
        self.epoch += 1

    def __len__(self):
        return len(self._blocks())


def fit_moments_stats(files, n_genes, moments_dir, block=8192, cache=None, force=False):
    """在 moments 值上拟合 HVG 词表与逐基因统计 (u, s)。"""
    if cache and os.path.exists(cache) and not force:
        z = np.load(cache, allow_pickle=True)
        if ("gene_names" in z.files and "schema_version" in z.files
                and int(np.asarray(z["schema_version"]).item()) >= 2
                and "requested_n_genes" in z.files
                and int(np.asarray(z["requested_n_genes"]).item()) == int(n_genes)):
            return (z["gene_names"], z["genes"], z["u_mu"], z["u_sd"],
                    z["s_mu"], z["s_sd"], z["file_hash"])
        z.close()

    from gfg3_data import file_vocabulary
    if not files:
        raise ValueError("cannot fit moments statistics from an empty split")
    # Moments are stored in each source file's vocabulary order.  Aggregate
    # into an explicit union panel instead of assuming that the first file is
    # representative.
    vocab_set = set()
    source_vocabs = []
    for p in files:
        vv = file_vocabulary(p)
        source_vocabs.append(vv)
        vocab_set.update(vv.tolist())
    v = np.array(sorted(vocab_set), dtype=str)
    G = len(v)
    target_index = {g: i for i, g in enumerate(v.tolist())}
    source_maps = [np.array([target_index[g] for g in vv], dtype=np.int64)
                   for vv in source_vocabs]
    S1 = np.zeros((2, G)); S2 = np.zeros((2, G)); cnt = 0
    rng = np.random.default_rng(7)
    sample = rng.permutation(len(files))[: min(len(files), 500)]
    for fi in sample:
        sid = os.path.basename(files[fi])[:-5]
        up_ = os.path.join(moments_dir, sid + ".u.npy")
        sp_ = os.path.join(moments_dir, sid + ".s.npy")
        if not (os.path.exists(up_) and os.path.exists(sp_)):
            continue
        U = np.load(up_, mmap_mode="r"); S = np.load(sp_, mmap_mode="r")
        n = U.shape[0]
        for r0 in range(0, n, block):
            r1 = min(r0 + block, n)
            u = np.asarray(U[r0:r1]); s = np.asarray(S[r0:r1])
            m = source_maps[fi]
            # add each source column to its union-vocabulary position; this is
            # cheap (G is only the metadata/panel axis) and handles reordering.
            np.add.at(S1[0], m, s.sum(0)); np.add.at(S2[0], m, (s ** 2).sum(0))
            np.add.at(S1[1], m, u.sum(0)); np.add.at(S2[1], m, (u ** 2).sum(0))
            cnt += r1 - r0
    mean = S1 / max(cnt, 1); var = np.maximum(S2 / max(cnt, 1) - mean ** 2, 0)
    order = np.argsort(-var[0]); order = [i for i in order if mean[0][i] > 0][:n_genes]
    genes = np.array(sorted(order))
    payload = dict(gene_names=v[genes], genes=genes,
                   u_mu=mean[1][genes], u_sd=np.sqrt(np.maximum(var[1][genes], 1e-8)),
                   s_mu=mean[0][genes], s_sd=np.sqrt(np.maximum(var[0][genes], 1e-8)),
                   file_hash="moments", schema_version=np.array(2, dtype=np.int64),
                   n_genes=np.array(len(genes), dtype=np.int64),
                   requested_n_genes=np.array(n_genes, dtype=np.int64))
    if cache:
        cache_dir = os.path.dirname(cache)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        np.savez(cache, **payload)
    return (payload["gene_names"], payload["genes"], payload["u_mu"], payload["u_sd"],
            payload["s_mu"], payload["s_sd"], payload["file_hash"])
