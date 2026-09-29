import numpy as np
import torch
from sklearn.decomposition import PCA
import os
import scvelo as scv
from preprocessing import *

def z_score(x, mean, std):
    if isinstance(mean, np.ndarray):
        mean = torch.tensor(mean, dtype=x.dtype, device=x.device if isinstance(x, torch.Tensor) else None)
    if isinstance(std, np.ndarray):
        std = torch.tensor(std, dtype=x.dtype, device=x.device if isinstance(x, torch.Tensor) else None)
    return (x - mean) / std

def re_z_score(x, mean, std):
    if isinstance(mean, np.ndarray):
        mean = torch.tensor(mean, dtype=x.dtype, device=x.device if isinstance(x, torch.Tensor) else None)
    if isinstance(std, np.ndarray):
        std = torch.tensor(std, dtype=x.dtype, device=x.device if isinstance(x, torch.Tensor) else None)
    return x * std + mean

def compute_pred(adata, dataloader, model, device='cuda'):
    u_preds, s_preds, tv_u_preds, tv_s_preds, v_u_preds, v_s_preds = [], [], [], [], [], []
        
    for i, batch in enumerate(dataloader):
        # 注意：这里假设 batch 是 (inputs, is_root) 元组，与训练一致
        if isinstance(batch, (list, tuple)):
            batch_inputs = batch[0]
            is_root = batch[1] if len(batch) > 1 else None
        else:
            batch_inputs = batch
            is_root = None  # 如果没有 is_root 标签，设为 None
            
        x_hat, tv = model.predict(batch_inputs, device=device)
        # 拆回 unspliced / spliced
        u_pred, s_pred = x_hat[:, :model.G], x_hat[:, model.G:]
        tv_u, tv_s = tv[:, :model.G], tv[:, model.G:]
        u_preds.append(u_pred.detach().cpu())
        s_preds.append(s_pred.detach().cpu())
        tv_u_preds.append(tv_u.detach().cpu())
        tv_s_preds.append(tv_s.detach().cpu())
        
    u_preds = torch.cat(u_preds, dim=0)
    s_preds = torch.cat(s_preds, dim=0)
    tv_u_preds = torch.cat(tv_u_preds, dim=0)
    tv_s_preds = torch.cat(tv_s_preds, dim=0)

    # 存入 adata
    adata.obsm["u_pred"] = u_preds.numpy()
    adata.obsm["s_pred"] = s_preds.numpy()
    adata.obsm["v_u_pred"] = tv_u_preds.numpy()
    adata.obsm["v_s_pred"] = tv_s_preds.numpy()
    adata.layers["velocity"] = tv_s_preds.numpy()  # (N_cells, N_genes)
    
    # adata.obsm["velocity_pca"] = adata.layers["velocity"] @ adata.varm["PCs"]
    
    
    # ========== 添加邻居图检查 ==========
    def check_and_fix_neighbors(adata, n_neighbors=30, n_pcs=30):
        """
        检查并修复邻居图
        """
        print("\n=== 开始检查邻居图 ===")
        
        # 1. 检查邻居图是否存在且有效
        neighbors_valid = False
        issues = []
        
        # 检查 uns 中是否有 neighbors
        if 'neighbors' not in adata.uns:
            issues.append("adata.uns 中缺少 'neighbors'")
        else:
            # 检查 neighbors 中是否有必要的键
            required_keys = ['params', 'connectivities_key', 'distances_key']
            for key in required_keys:
                if key not in adata.uns['neighbors']:
                    issues.append(f"adata.uns['neighbors'] 中缺少 '{key}'")
            
            # 检查 params 中的关键参数
            if 'params' in adata.uns['neighbors']:
                params = adata.uns['neighbors']['params']
                if 'n_neighbors' not in params:
                    issues.append("params 中缺少 'n_neighbors'")
                if 'method' not in params:
                    issues.append("params 中缺少 'method'")
        
        # 检查 obsp 中的距离矩阵和连接矩阵
        if 'distances' not in adata.obsp:
            issues.append("adata.obsp 中缺少 'distances'")
        else:
            # 检查距离矩阵是否有效
            distances = adata.obsp['distances']
            if distances.shape[0] != adata.n_obs or distances.shape[1] != adata.n_obs:
                issues.append(f"distances 形状错误: {distances.shape}，期望: ({adata.n_obs}, {adata.n_obs})")
            elif distances.nnz == 0:
                issues.append("distances 矩阵为空（没有非零元素）")
            else:
                # kNN 距离矩阵通常是有向的；不对称本身不是损坏。
                if not np.isfinite(distances.data).all():
                    issues.append("distances 包含 NaN 或 Inf")
                elif (distances.data < 0).any():
                    issues.append("distances 包含负值")
                else:
                    neighbors_valid = True
        
        if 'connectivities' not in adata.obsp:
            issues.append("adata.obsp 中缺少 'connectivities'")
        else:
            connectivities = adata.obsp['connectivities']
            if connectivities.shape[0] != adata.n_obs or connectivities.shape[1] != adata.n_obs:
                issues.append(f"connectivities 形状错误: {connectivities.shape}，期望: ({adata.n_obs}, {adata.n_obs})")
            elif connectivities.nnz == 0:
                issues.append("connectivities 矩阵为空")
        
        # 2. 输出检查结果
        print(f"邻居图有效性: {'✅ 有效' if neighbors_valid else '❌ 无效'}")
        if issues:
            print("发现的问题:")
            for issue in issues:
                print(f"  ⚠️ {issue}")
        else:
            print("✅ 所有检查通过")
        
        # 3. 如果无效，强制重建
        if not neighbors_valid or issues:
            print("\n🔄 开始重建邻居图...")
            
            # 彻底清理旧的邻居图
            print("  清理旧邻居图...")
            keys_to_remove = []
            for key in adata.uns.keys():
                if 'neighbors' in key.lower():
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                print(f"    删除 adata.uns['{key}']")
                del adata.uns[key]
            
            for key in list(adata.obsp.keys()):
                if key in ['distances', 'connectivities']:
                    print(f"    删除 adata.obsp['{key}']")
                    del adata.obsp[key]
            
            # 重新计算邻居图
            print("  重新计算邻居图...")
            import scanpy as sc
            try:
                # 先尝试用 scanpy
                sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=42)
                print(f"  使用 scanpy 成功计算邻居图 (n_neighbors={n_neighbors}, n_pcs={n_pcs})")
            except Exception as e:
                print(f"  scanpy 计算失败: {e}")
                try:
                    # 如果 scanpy 失败，尝试用 scvelo
                    import scvelo as scv
                    scv.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
                    print(f"  使用 scvelo 成功计算邻居图 (n_neighbors={n_neighbors}, n_pcs={n_pcs})")
                except Exception as e2:
                    print(f"  scvelo 计算也失败: {e2}")
                    raise ValueError("无法重建邻居图，请检查数据格式")
            
            # 验证重建结果
            print("  验证重建结果...")
            if 'distances' in adata.obsp and 'connectivities' in adata.obsp:
                print(f"    distances shape: {adata.obsp['distances'].shape}")
                print(f"    connectivities shape: {adata.obsp['connectivities'].shape}")
                print("  ✅ 邻居图重建成功")
            else:
                raise ValueError("重建后邻居图仍然不完整")
             
        else:
            print("\n✅ 邻居图有效，跳过重建")

        adata = build_neighbor_indices(adata, n_neighbors=n_neighbors)
        return adata
    
    # ========== 执行检查和修复 ==========
    adata = check_and_fix_neighbors(adata, n_neighbors=30, n_pcs=30)
    
    # ========== 原有的计算逻辑 ==========
    print("\n开始计算 velocity_graph...")
    scv.tl.velocity_graph(adata, vkey="velocity")

    return adata
    
def compute_latent(adata, dataloader, model, device='cuda'):
    import scanpy as sc
    """
    计算模型隐空间 latent_z，并存储 latent_presentation 和 latent_umap

    参数:
        adata: AnnData 对象
        model: 已训练的 VeloModel
        latent_name: 存储 latent 表示的 obsm 键名
        umap_name: 存储 latent UMAP 的 obsm 键名
        device: 'cuda' 或 'cpu'
        n_neighbors, n_components, min_dist, random_state: UMAP 参数
    """
    latent_name = "latent_presentation"
    umap_name = "latent_umap"
    pca_name = "latent_pca"
    
    n_neighbors = model.config.config["data"]['num_neighbor']
    
    latent_z_list = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if isinstance(batch, (list, tuple)):
                batch_inputs = batch[0]
                is_root = batch[1] if len(batch) > 1 else None
            else:
                batch_inputs = batch
                is_root = None  # 如果没有 is_root 标签，设为 None
            
            latent_z = model.get_latent(batch_inputs, device=device)  # (B, G * D)
            latent_z_list.append(latent_z.cpu())

    # 拼接所有批次的 latent_z
    latent_z = torch.cat(latent_z_list, dim=0)  # (N, G * D)

    pca = PCA(n_components=30, random_state=0)
    adata.obsm[latent_name] = pca.fit_transform(latent_z.cpu().numpy())  # 保存 latent_presentation
    
    # 使用 latent 表示计算 PCA
    tmp_adata = sc.AnnData(adata.obsm[latent_name])
    sc.tl.pca(tmp_adata, n_comps=2, random_state=42)
    adata.obsm[pca_name] = tmp_adata.obsm['X_pca']

    # 使用 latent 表示计算 UMAP
    tmp_adata = sc.AnnData(adata.obsm[latent_name])
    sc.pp.neighbors(tmp_adata, n_neighbors=n_neighbors, use_rep='X')
    sc.tl.umap(tmp_adata, n_components=2, min_dist=2, spread=6.0, random_state=42)
    adata.obsm[umap_name] = tmp_adata.obsm['X_umap']
    
    adata.obsm["X_ours"] = adata.obsm[umap_name]
    
    print(f"隐表示已保存到 adata.obsm['{latent_name}']，UMAP 已保存到 adata.obsm['{umap_name}']")
    return adata

def overlap_name(train_data, pred_data, ignore=True):
    """
    返回筛选后 train_data 和 pred_data，它们包含相同的交集基因（顺序一致）。
    可选择忽略大小写。
    """
    # 获取基因名集合
    if ignore:
        genes_train = set(train_data.var.index.str.upper())
        genes_pred = set(pred_data.var.index.str.upper())
    else:
        genes_train = set(train_data.var.index)
        genes_pred = set(pred_data.var.index)

    # 计算交集并保持 train_data 顺序
    if ignore:
        common_genes = [gene for gene in train_data.var.index if gene.upper() in genes_pred]
    else:
        common_genes = [gene for gene in train_data.var.index if gene in genes_pred]

    if len(common_genes) == 0:
        raise ValueError("train_data 和 pred_data 之间没有共同基因，请检查基因命名是否一致（如大小写、符号等）。")
    else:
        print(f"交集基因数为：{len(common_genes)}")

    # 按交集基因筛选 pred_data
    if ignore:
        upper_to_original_pred = {g.upper(): g for g in pred_data.var.index}
        upper_to_original_train = {g.upper(): g for g in train_data.var.index}

        matched_genes_pred = [upper_to_original_pred[g.upper()] for g in common_genes if g.upper() in upper_to_original_pred]
        matched_genes_train = [upper_to_original_train[g.upper()] for g in common_genes if g.upper() in upper_to_original_train]
    else:
        matched_genes_pred = common_genes
        matched_genes_train = common_genes

    # 保证两个 AnnData 的基因顺序一致
    train_filtered = train_data[:, matched_genes_train].copy()
    pred_filtered = pred_data[:, matched_genes_pred].copy()

    return train_filtered, pred_filtered

def load_model(model, model_path, device='cuda'):
    """
    加载保存的模型参数到 VeloModel 实例。

    参数:
        model: VeloModel 实例，未加载参数
        model_path: str, 保存的模型参数文件路径（.pth 文件）
        device: str, 设备（'cuda' 或 'cpu'）

    返回:
        model: VeloModel 实例，已加载参数
    """
    print("加载模型")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()  # 设置为评估模式
    print(f"模型参数已从 {model_path} 加载")
    return model
    
def find_cluster_key(adata, cluster_key=None):
    if cluster_key is None:
        cluster_label = ['Annotation', 'cell_type', 'clusters', 'Clusters', 'cluster', 'Cluster', 'phase', 'celltype', 'Celltype']
        cluster_key = next((s for s in cluster_label if s in adata.obs), None)
    if cluster_key is None:
        print("未找到聚类标签，可用", adata.obs)  # 如果未找到聚类标签，打印警告
    else: 
        cluster_labels = adata.obs[cluster_key].astype('category').cat.codes.values if cluster_key else None
        cluster_labels = torch.tensor(cluster_labels, dtype=torch.long) if cluster_labels is not None else None
        print(f"共有{len(adata.obs[cluster_key].unique())}种{cluster_key}标签：{adata.obs[cluster_key].unique()}")
    
    return cluster_key, cluster_labels

def concatenate_adata(adata1, adata2, ignore_case=True):
    """
    拼接两个adata对象，只保留unspliced、spliced和cluster信息
    
    参数:
        adata1: AnnData对象，作为参考数据集
        adata2: AnnData对象，作为待拼接的数据集
        cluster_key: str或None，聚类标签的键名。如果为None，自动查找
        ignore_case: bool，是否在取基因交集时忽略大小写
    
    返回:
        combined_adata: 拼接后的AnnData对象
    """
    from scipy import sparse
    import scanpy as sc
    # 2. 取基因交集
    adata1_common, adata2_common = overlap_name(adata1, adata2, ignore=ignore_case)
    
    cluster_key1, _ = find_cluster_key(adata1)
    cluster_key2, _ = find_cluster_key(adata2)
    
    # 4. 提取所需的数据
    def extract_data(adata, common_genes_idx, cluster_key):
        # 提取unspliced和spliced数据（仅保留共同基因）
        unspliced = adata.layers['unspliced'][:, common_genes_idx]
        spliced = adata.layers['spliced'][:, common_genes_idx]
        
        # 提取cluster标签
        clusters = adata.obs[cluster_key].values
        X = spliced.copy()
        
        return unspliced, spliced, X, clusters
    
    # 获取共同基因的索引
    common_genes = adata1_common.var.index.tolist()
    idx1 = [adata1.var.index.get_loc(gene) for gene in common_genes]
    idx2 = [adata2.var.index.get_loc(gene) for gene in common_genes]
    
    # 提取数据
    unspliced1, spliced1, X1, clusters1 = extract_data(adata1, idx1, cluster_key1)
    unspliced2, spliced2, X2, clusters2 = extract_data(adata2, idx2, cluster_key2)
    
    # 5. 创建新的AnnData对象
    # 合并基因信息
    var_df = adata1_common.var.copy()
    
    # 合并观测信息
    obs_dict = {
        'dataset': ['dataset1'] * adata1.n_obs + ['dataset2'] * adata2.n_obs,
        cluster_key1: np.concatenate([clusters1, clusters2])
    }
    
    # 创建新的AnnData
    combined_adata = sc.AnnData(
        X=sparse.vstack([X1, X2]), 
        obs=obs_dict,
        var=var_df
    )
    
    # 添加layers
    combined_adata.layers['unspliced'] = sparse.vstack([unspliced1, unspliced2])
    combined_adata.layers['spliced'] = sparse.vstack([spliced1, spliced2])
    
    # 6. 打印统计信息
    print(f"拼接完成!")
    print(f"  数据集1细胞数: {adata1.n_obs}")
    print(f"  数据集2细胞数: {adata2.n_obs}")
    print(f"  共同基因数: {len(common_genes)}")
    print(f"  总细胞数: {combined_adata.n_obs}")
    print(f"  cluster键名: {cluster_key1}")
    print(f"  cluster类别数: {len(np.unique(combined_adata.obs[cluster_key1]))}")
    
    return combined_adata