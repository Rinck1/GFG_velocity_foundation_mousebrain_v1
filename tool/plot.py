import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import seaborn as sns
import torch
import scipy as sp
import scvelo as scv
def plot_velocity_stream(adata, vkey, basis, cluster_key, save_path="figures/velocity_stream.png"):
    
    import matplotlib.pyplot as plt
    from adjustText import adjust_text
    print("绘制速度图")
    fig, ax = plt.subplots(figsize=(6, 6))
    scv.pl.velocity_embedding_stream(
        adata,
        basis='umap',
        color=cluster_key,
        legend_fontsize = 12,
        fontsize=12,
        ax=ax,
        show=False
    )
    adjust_text(ax.texts)
    # np.random.seed(0)
    # data = np.load("label_pos.npz")
    # label_pos = {k: tuple(data[k]) for k in data.files}
    # offset = 0.6  # 左移量
    # for t in ax.texts:
    #     txt = t.get_text()
    #     if txt in label_pos:
    #         x, y = label_pos[txt]
    #         t.set_position((x - offset, y))
    
    # import matplotlib.patches as mpatches
    # clusters = adata.obs[cluster_key].cat.categories
    # palette = scv.pl.palettes.default_20[:len(clusters)]
    # handles = [
    #     mpatches.Patch(color=palette[i], label=cl)
    #     for i, cl in enumerate(clusters)
    # ]
    # ax.legend(
    #     handles=handles,
    #     loc='upper right',
    #     fontsize=8,
    #     frameon=True,
    #     title='Clusters',
    #     ncol=2
    # )
    
    ax.set_title("GFG (Ours)",fontsize=14,fontweight="bold")
    
    plt.savefig("figures/velocity_stream.png",dpi=300,bbox_inches="tight")
    print("速度图已保存到 figures/velocity_stream.png")
    
def plot_s_v_distribution(adata, bins=200, save_path=None, quantile=0.95, log1p=False):
    """
    左图: s_pred vs spliced (对比直方图)
    右图: v_pred 分布
    """
    def to_dense(arr):
        """将稀疏矩阵转换为稠密，兼容 ndarray"""
        if sp.sparse.issparse(arr):
            return arr.toarray()
        return np.array(arr)

    s_pred = to_dense(adata.obsm['s_pred']).flatten()
    v_pred = to_dense(adata.obsm['v_s_pred']).flatten()
    s_true = to_dense(adata.layers['spliced']).flatten()
    
    if log1p:
        print("log1p")
        s_pred = np.log1p(s_pred)
        s_true = np.log1p(s_true)

    def central_values(arr, q=quantile):
        """保留中心 q 部分的数据，去掉极端值"""
        lower = (1 - q) / 2 * 100
        upper = (1 + q) / 2 * 100
        low_val, high_val = np.percentile(arr, [lower, upper])
        return arr[(arr >= low_val) & (arr <= high_val)]

    s_pred_central = central_values(s_pred, quantile)
    v_pred_central = central_values(v_pred, quantile)
    s_true_central = central_values(s_true, quantile)

    # ---------- 绘图 ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 左边: s_pred vs spliced
    ax1.hist(s_true_central, bins=bins, density=True, alpha=0.5, color="gray", label="spliced (true)")
    ax1.hist(s_pred_central, bins=bins, density=True, alpha=0.5, color="blue", label="s_pred")
    ax1.set_title(f"spliced vs s_pred (central {int(quantile*100)}%)")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Density")
    ax1.legend()

    # 右边: v_pred
    ax2.hist(v_pred_central, bins=bins, density=True, alpha=0.7, color="green")
    ax2.set_title(f"Distribution of v_pred (central {int(quantile*100)}%)")
    ax2.set_xlabel("Value")
    ax2.set_ylabel("Density")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_latent(latent, clusters, save_path="figures/latent.png"):
    """
    latent_z: (n_samples, 2)
    clusters: (n_samples,)
    """

    # 转成 DataFrame
    DF = pd.DataFrame({
        "z1": latent[:, 0],
        "z2": latent[:, 1],
        "cluster": clusters
    })

    # ---------- 绘图 ----------
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=DF, x="z1", y="z2", hue="cluster",
                    palette="tab10", s=10, legend="full")
    plt.title("Latent space (PCA)")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

def plot_codebook_usage(m_usage, v_usage, save_path="figures/codebook_usage.png"):
    """
    绘制硬分配码本使用情况（概率或计数）。

    参数：
        m_usage (Tensor 或 np.ndarray): 流形码本使用情况，一维数组 [codebook_size]。
        v_usage (Tensor 或 np.ndarray): 速度码本使用情况，一维数组 [codebook_size]。
        save_path (str, optional): 保存路径。
    """
    # 转换为 numpy
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return x

    m_usage = to_numpy(m_usage)
    v_usage = to_numpy(v_usage)

    codebook_indices = np.arange(len(m_usage))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 流形码本
    axes[0].bar(codebook_indices, m_usage, color='blue', alpha=0.7)
    axes[0].set_title("Manifold Codebook Usage")
    axes[0].set_xlabel("Codeword Index")
    axes[0].set_ylabel("Usage Count / Probability")
    axes[0].set_xticks(codebook_indices)

    # 速度码本
    axes[1].bar(codebook_indices, v_usage, color='red', alpha=0.7)
    axes[1].set_title("Velocity Codebook Usage")
    axes[1].set_xlabel("Codeword Index")
    axes[1].set_ylabel("Usage Count / Probability")
    axes[1].set_xticks(codebook_indices)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"码本使用情况柱状图已保存到 {save_path}")
    else:
        plt.show()

def plot_loss_curves(loss_history, save_path="figures/loss_curve.png"):
    if loss_history is None:
        return
    
    keys = list(loss_history.keys())
    num_keys = len(keys)

    ncols = 3
    nrows = (num_keys + ncols - 1) // ncols
    plt.figure(figsize=(5 * ncols, 4 * nrows))

    for i, key in enumerate(keys):
        plt.subplot(nrows, ncols, i + 1)
        # 过滤掉 None
        values = [v for v in loss_history[key] if v is not None]
        plt.plot(values, label=key, color='tab:blue')
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.title(key)
        plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss 曲线已保存至 {save_path}")

def plot_manifold_codebook(model, top_k=3, save_path=None):
    """
    并排绘制流形码本和速度码本 (使用球面切空间 PCA)，
    自动选择码本方差最大的基因，并调整可视范围。

    参数:
        model: 含有 model.manifold_codebook / model.velocity_codebook
        top_k: 画方差最大的前多少个基因
        save_path: 如果指定，比如 "figures/codebooks.png"，会保存一张图
    """
    def _frechet_mean_on_sphere(X, iters=50, eps=1e-9):
        mu = X.mean(axis=0)
        mu = mu / (np.linalg.norm(mu) + eps)
        for _ in range(iters):
            dots = np.clip(X @ mu, -1.0, 1.0)
            thetas = np.arccos(dots)
            sins = np.sin(thetas) + eps
            tangents = ((thetas / sins)[:, None]) * (X - (dots[:, None]) * mu[None, :])
            v = tangents.mean(axis=0)
            if np.linalg.norm(v) < 1e-8:
                break
            normv = np.linalg.norm(v)
            dirv = v / (normv + eps)
            mu = np.cos(normv) * mu + np.sin(normv) * dirv
            mu = mu / (np.linalg.norm(mu) + eps)
        return mu

    def spherical_tangent_pca(X, base_mu=None):
        eps = 1e-9
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + eps)
        mu = base_mu if base_mu is not None else _frechet_mean_on_sphere(X)
        dots = np.clip(X @ mu, -1.0, 1.0)
        thetas = np.arccos(dots)
        sins = np.sin(thetas) + eps
        V = ((thetas / sins)[:, None]) * (X - (dots[:, None]) * mu[None, :])
        Vc = V - V.mean(axis=0, keepdims=True)
        U, S, Vh = np.linalg.svd(Vc, full_matrices=False)
        Y2d = Vc @ Vh.T[:, :2]
        return Y2d, mu

    def prepare_embeddings(embeddings):
        n_genes = embeddings.shape[0]
        emb_flat = embeddings.reshape(-1, embeddings.shape[-1])
        emb_flat = emb_flat / np.linalg.norm(emb_flat, axis=1, keepdims=True)
        emb_2d, mu = spherical_tangent_pca(emb_flat)
        codes_per_gene = embeddings.shape[1]
        return emb_2d, n_genes, codes_per_gene

    def select_top_genes(embeddings, top_k):
        # 计算每个基因的码本方差
        variances = embeddings.var(axis=(1, 2))
        top_genes = np.argsort(-variances)[:top_k]
        return top_genes

    # 获取码本
    manifold_embeddings = model.manifold_codebook.get_embedding().detach().cpu().numpy()
    velocity_embeddings = model.velocity_codebook.get_embedding().detach().cpu().numpy()

    manifold_2d, n_genes, codes_per_gene = prepare_embeddings(manifold_embeddings)
    velocity_2d, _, _ = prepare_embeddings(velocity_embeddings)

    # 挑选方差最大的基因
    top_genes_manifold = select_top_genes(manifold_embeddings, top_k)
    top_genes_velocity = select_top_genes(velocity_embeddings, top_k)

    # 调色盘
    colors = plt.cm.tab10(np.linspace(0, 1, top_k))

    # 绘图 (左右两个子图)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

    for ax, emb_2d, title, top_genes in zip(
        axes,
        [manifold_2d, velocity_2d],
        ["Manifold Codebook (tangent PCA)", "Velocity Codebook (tangent PCA)"],
        [top_genes_manifold, top_genes_velocity]
    ):
        for i, g in enumerate(top_genes):
            idx_start = g * codes_per_gene
            idx_end = (g + 1) * codes_per_gene
            ax.scatter(
                emb_2d[idx_start:idx_end, 0],
                emb_2d[idx_start:idx_end, 1],
                alpha=0.7,
                label=f"Gene {g}",
                color=colors[i % len(colors)]
            )
        ax.set_xlabel("PC1 (tangent space)")
        ax.set_ylabel("PC2 (tangent space)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        ax.set_aspect("equal", "box")
        ax.autoscale()

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
        
def plot_paga(adata, basis="X_umap", cluster_key="cluster", save_path="figures/PAGA.png", title="PAGA"):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import networkx as nx
    from sklearn.neighbors import NearestNeighbors
    import os

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ========== 1. 基础数据 ==========
    umap = adata.obsm[basis]
    vel_umap = adata.obsm["velocity_umap"]
    clusters = adata.obs[cluster_key].astype(str).values

    df = pd.DataFrame({"x": umap[:,0], "y": umap[:,1], "cluster": clusters})
    unique_clusters = sorted(df["cluster"].unique())

    # ========== 2. 自动分 cluster 颜色 ==========
    cmap = plt.get_cmap("tab20")
    colors = {cl: cmap(i / len(unique_clusters)) for i, cl in enumerate(unique_clusters)}

    # ========== 3. cluster center ==========
    centers = df.groupby("cluster")[["x","y"]].mean()

    # ========== 4. cluster velocity mean ==========
    cluster_velocity = {}
    for cl in unique_clusters:
        mask = df["cluster"] == cl
        cluster_velocity[cl] = vel_umap[mask].mean(axis=0)[:2]  # 2D velocity

    # ========== 5. 建 cluster KNN 图 (代替 PAGA 图) ==========
    nbrs = NearestNeighbors(n_neighbors=3).fit(centers.values)
    _, indices = nbrs.kneighbors(centers.values)

    G = nx.Graph()
    for c in centers.index:
        G.add_node(c)
    for i, c in enumerate(centers.index):
        for j in indices[i][1:]:
            G.add_edge(c, centers.index[j])

    # ========== 6. 绘图 ==========
    fig, ax = plt.subplots(figsize=(10, 8))

    # 背景点
    for cl in unique_clusters:
        sub = df[df["cluster"] == cl]
        ax.scatter(sub["x"], sub["y"], s=30, color=colors[cl], alpha=0.45)

    # cluster 中心 + label
    for cl in unique_clusters:
        x, y = centers.loc[cl]
        ax.scatter(x, y, s=160, color=colors[cl])
        ax.text(x, y, cl, fontsize=17, weight="bold", ha="center", va="center")

    # ========== 7. 画 cluster 间的直线箭头 ==========
    for u, v in G.edges():
        x1, y1 = centers.loc[u]
        x2, y2 = centers.loc[v]

        # cluster u 的 velocity（决定方向力度）
        vx, vy = cluster_velocity[u]

        # 用 velocity 调整中间目标点的位置
        #（你可以把这个设为 0，表示完全不考虑 velocity，只画 straight line）
        mid_x = x1 + vx * 0.6
        mid_y = y1 + vy * 0.6

        # 线段：u → v（不再 spline）
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=2.0, alpha=0.8)

        # 线段中点
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2

        # 箭头长度（占整条线的比例，可调）
        arrow_scale = 0.08  # 越小箭头越短

        dx = x2 - x1
        dy = y2 - y1

        # 箭头起点 → 终点（都在中点附近）
        arrow_start_x = mx - dx * arrow_scale
        arrow_start_y = my - dy * arrow_scale
        arrow_end_x   = mx + dx * arrow_scale
        arrow_end_y   = my + dy * arrow_scale

        ax.annotate(
            "",
            xy=(arrow_end_x, arrow_end_y),
            xytext=(arrow_start_x, arrow_start_y),
            arrowprops=dict(arrowstyle="-|>", lw=5, color="black")
        )

    ax.set_title(title, fontsize=15)
    ax.axis("off")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"PAGA saved to: {save_path}")