import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def ploting(adata, cluster_key, gene=None, save_path="figures/check_data.png"):
    """
    将所有 cluster 的 Unspliced vs Spliced 分布图绘制在一张大图中，
    并统一坐标范围（带 padding，防止图形被裁剪）。
    """

    # 准备 unspliced / spliced 数据
    if gene is None:
        unspliced = adata.layers["unspliced"].sum(axis=1).A1
        spliced   = adata.layers["spliced"].sum(axis=1).A1
    else:
        g_idx = adata.var_names.get_loc(gene)
        unspliced = adata.layers["unspliced"][:, g_idx].A1
        spliced   = adata.layers["spliced"][:, g_idx].A1

    df = pd.DataFrame({
        "cluster": adata.obs[cluster_key].astype(str),
        "unspliced": unspliced,
        "spliced": spliced,
    })

    clusters = sorted(df["cluster"].unique())
    n = len(clusters)

    # -------- 关键：统一坐标轴，但加 padding 防止 KDE 被切掉 --------
    xmin, xmax = df["unspliced"].min(), df["unspliced"].max()
    ymin, ymax = df["spliced"].min(),   df["spliced"].max()

    # padding = 5% 范围
    xpad = (xmax - xmin) * 0.05
    ypad = (ymax - ymin) * 0.05

    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    # 行列布局
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 4*nrows))
    axes = axes.flatten()

    # 绘图
    for i, c in enumerate(clusters):
        ax = axes[i]
        sub = df[df.cluster == c]

        sns.kdeplot(
            data=sub,
            x="unspliced",
            y="spliced",
            fill=True,
            cmap="viridis",
            ax=ax,
            thresh=0.02,   # 更保守一些，不容易被裁掉
        )

        ax.set_title(f"{c}", fontsize=12)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    # 隐藏多余子图
    for j in range(len(clusters), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"图已保存至：{save_path}")
        
def plot_cluster_graph(adata, cluster_key, cluster_edges, gene=None,
                       save_path="figures/cluster_graph.png"):
    """
    绘制 cluster 中心点，并根据 cluster_edges 画箭头，坐标自动调整。
    """
    # ------ 准备 unspliced / spliced 数据 ------
    if gene is None:
        unspliced = adata.layers["unspliced"].sum(axis=1).A1
        spliced   = adata.layers["spliced"].sum(axis=1).A1
    else:
        g_idx = adata.var_names.get_loc(gene)
        unspliced = adata.layers["unspliced"][:, g_idx].A1
        spliced   = adata.layers["spliced"][:, g_idx].A1

    df = pd.DataFrame({
        "cluster": adata.obs[cluster_key].astype(str),
        "unspliced": unspliced,
        "spliced": spliced,
    })

    clusters = sorted(df["cluster"].unique())

    # ------- 计算 cluster 中心 -------
    centers = {}
    for c in clusters:
        sub = df[df.cluster == c]
        centers[c] = (sub["unspliced"].mean(), sub["spliced"].mean())

    # ------- 自动确定坐标范围 -------
    all_x = [x for x, y in centers.values()]
    all_y = [y for x, y in centers.values()]
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = min(all_y), max(all_y)
    xpad = (xmax - xmin) * 0.1 if xmax > xmin else 1.0
    ypad = (ymax - ymin) * 0.1 if ymax > ymin else 1.0

    xmin -= xpad
    xmax += xpad
    ymin -= ypad
    ymax += ypad

    # ------- 绘图 -------
    fig, ax = plt.subplots(figsize=(7, 7))

    # 绘制中心点
    for c, (cx, cy) in centers.items():
        ax.scatter(cx, cy, s=120, alpha=0.9)
        ax.text(cx, cy, f" {c}", fontsize=10, va="center")

    # 绘制箭头
    for start, end in cluster_edges:
        if start not in centers or end not in centers:
            continue
        x0, y0 = centers[start]
        x1, y1 = centers[end]
        ax.annotate("",
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=2, alpha=0.8)
                    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Unspliced")
    ax.set_ylabel("Spliced")
    ax.set_title("Cluster Center Graph (Auto coordinates)")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"cluster graph 已保存至：{save_path}") 
if __name__ == "__main__":
    from model.Config import Config
    from preprocessing import * 
    import os
    from tool.utils import * 
    from const.cluster_edges import all_edges
    datasets = [
        "DentateGyrus.h5ad",
        "Pancreas.h5ad",
        "MouseBrain.h5ad",
        "endocrinogenesis_day15.h5ad", 
        "Hindbrain_GABA_Glio.h5ad",
        "erythroid_lineage.h5ad"
    ]
    data_index = 0
    adata = load_data("data/" + datasets[data_index])
        
    # train_adata = load_data("data/" + datasets[3])
    # _, adata = overlap_name(train_adata, adata)
    
    config = Config(adata,seed=0)
    cluster_key, cluster_labels = find_cluster_key(adata)
    cluster_edges = all_edges.get(datasets[data_index])
    ploting(adata, cluster_key=cluster_key, save_path="figures/check_data(pre).png")
    plot_cluster_graph(
        adata,
        cluster_key=cluster_key,
        cluster_edges=cluster_edges,
        save_path="figures/cluster_graph(pre).png"
    )
    adata = preprocess_data_scv(
        adata, 
        n_top_genes=config.config["data"]["num_top_gene"], 
        n_neighbors=config.config["data"]["num_neighbor"]
    )
    ploting(adata, cluster_key=cluster_key)
    plot_cluster_graph(
        adata,
        cluster_key=cluster_key,
        cluster_edges=cluster_edges,
        save_path="figures/cluster_graph.png"
    )