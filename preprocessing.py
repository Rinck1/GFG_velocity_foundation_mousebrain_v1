import scanpy as sc
import scvelo as scv
import numpy as np
import anndata as ad

def load_data(path):
    adata = sc.read(path, cache=True)
    
    # 输出基本信息
    print(f"数据已加载：")
    print(f"  细胞数: {adata.n_obs}")
    print(f"  基因数: {adata.n_vars}")
    
    print("\n=== 检查 obs ===")
    print(f"列名: {adata.obs.columns.tolist()}")
    
    print("\n=== 检查 obsm ===")
    print(f"列名: {list(adata.obsm.keys())}")
    
    print("\n=== 检查 layers ===")
    if adata.layers:
        print(f"层名: {list(adata.layers.keys())}")
        for k in adata.layers.keys():
            print(f"  {k} 形状: {adata.layers[k].shape}")
    else:
        print("layers 为空，请检查 check_layers 函数是否成功添加了层。")
        
    adata = check_layers(adata)
    return adata

def clean_nan_inf(adata: ad.AnnData) -> ad.AnnData:
    # ---- 清理 .X ----
    data = adata.X
    if hasattr(data, "toarray"):
        data = data.toarray()
    adata.X = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    print("[clean_nan_inf] 清洗完成: X")

    # ---- 清理所有 layers ----
    for layer_name, layer_data in adata.layers.items():
        if hasattr(layer_data, "toarray"):
            layer_data = layer_data.toarray()
        adata.layers[layer_name] = np.nan_to_num(layer_data, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"[clean_nan_inf] 清洗完成: {layer_name}")

    return adata

def build_neighbor_indices(adata, n_neighbors=30, n_pcs=30):
    """Store actual sparse-distance neighbors, padding short rows with -1."""
    if "distances" not in adata.obsp:
        raise KeyError("adata.obsp['distances'] is required to build neighbor indices")

    distances = adata.obsp["distances"].tocsr()
    indices = np.full((adata.n_obs, n_neighbors), -1, dtype=np.int64)

    for cell_index in range(adata.n_obs):
        row_start = distances.indptr[cell_index]
        row_end = distances.indptr[cell_index + 1]
        neighbors = distances.indices[row_start:row_end]
        neighbor_distances = distances.data[row_start:row_end]

        valid = (
            (neighbors != cell_index)
            & np.isfinite(neighbor_distances)
            & (neighbor_distances > 0)
        )
        neighbors = neighbors[valid]
        neighbor_distances = neighbor_distances[valid]

        if neighbors.size:
            order = np.argsort(neighbor_distances)
            selected = neighbors[order[:n_neighbors]]
            indices[cell_index, :selected.size] = selected

    adata.uns["neighbors"]["indices"] = indices
    return adata

def check_layers(adata):
    layers = adata.layers.keys()
    
    # 如果没有 'spliced'，但有 labeled/unlabeled，则自动合并
    if 'spliced' not in layers:
        if 'labeled_spliced' in layers and 'unlabeled_spliced' in layers:
            adata.layers['spliced'] = adata.layers['labeled_spliced'] + adata.layers['unlabeled_spliced']
            print("构建 'spliced' 层（labeled + unlabeled spliced）")
    if 'unspliced' not in layers:
        if 'labeled_unspliced' in layers and 'unlabeled_unspliced' in layers:
            adata.layers['unspliced'] = adata.layers['labeled_unspliced'] + adata.layers['unlabeled_unspliced']
            print("构建 'unspliced' 层（labeled + unlabeled unspliced）")
    
    return adata
  
def add_root_mask(adata, root_cluster='NMP', cluster_key='cluster'):
    """
    一行调用，自动把 root 细胞存到 adata.obs['is_root']，并返回 mask
    """
    # 自动搜常见聚类列
    mask = adata.obs[cluster_key].astype(str) == str(root_cluster)
    if mask.sum() > 0:
        adata.obs['is_root'] = mask
        print(f"Found {mask.sum()} root cells: {cluster_key} == '{root_cluster}'")
        return adata, mask.values
    
    # 没找到就用 unspliced 比例最高的前 50 个细胞
    u = adata.layers['unspliced'].sum(1)
    s = adata.layers['spliced'].sum(1)
    ratio = u / (u + s + 1e-8)
    if hasattr(ratio, 'A1'): 
        ratio = ratio.A1
    top_idx = ratio.argsort()[-50:][::-1]  # 降序前50
    mask = np.zeros(adata.n_obs, dtype=bool)
    mask[top_idx] = True
    
    adata.obs['is_root'] = mask
    print("No cluster name matched, used top-50 highest u/(u+s) cells as root")
    return adata, mask          

def preprocess_data_scv(adata, n_top_genes=None, n_neighbors=30, seed=42):
            
    # scVelo 标准预处理
    scv.pp.remove_duplicate_cells(adata)
    filter_kwargs = {"min_shared_counts": 20}
    if n_top_genes is not None:
        filter_kwargs["n_top_genes"] = n_top_genes
    scv.pp.filter_and_normalize(adata, **filter_kwargs)
    
    sc.tl.pca(adata, n_comps=30, random_state=seed)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=30)
    scv.pp.moments(adata)
    sc.tl.umap(adata, random_state=seed)
    
    adata = build_neighbor_indices(adata, n_neighbors=n_neighbors)
    
    return adata