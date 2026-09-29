import numpy as np
import torch
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC

def evaluate_coherence(adata, vkey="velocity", eps=1e-9):
    """
    使用余弦相似度矩阵 + KNN 图计算每个细胞的 velocity coherence
    """
    # 获取 velocity 矩阵
    V = adata.layers[vkey]
    if isinstance(V, np.ndarray):
        V = torch.tensor(V, dtype=torch.float32)
    elif torch.is_tensor(V):
        pass
    else:
        V = torch.tensor(V.toarray(), dtype=torch.float32)

    if V.dim() == 3:
        V = V.squeeze(-1)
    
    V_norm = V / (V.norm(dim=1, keepdim=True) + eps)

    cos_sim_matrix = V_norm @ V_norm.T 

    G = adata.obsp["connectivities"] 
    if not isinstance(G, torch.Tensor):
        G = torch.tensor(G.toarray(), dtype=torch.float32)

    G = G / (G.sum(dim=1, keepdim=True) + eps) 
    coherence = (G * cos_sim_matrix).sum(dim=1)
    adata.obs["velocity_coherence"] = coherence.numpy().flatten()
    return coherence

def evaluate_state_prediction(adata, label_key, latent):
    if label_key not in adata.obs:
        print(f"'{label_key}' 不在 adata.obs 中。可用标签有：{adata.obs.columns.tolist()}")
        return None

    labels = adata.obs[label_key].astype(str).values
    X_train, X_test, y_train, y_test = train_test_split(latent, labels, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"使用 latent representation 预测 '{label_key}' 的准确率: {acc:.4f}")
    return acc

def evaluate_state_prediction_nonlinear(adata, label_key, latent):
    """
    用非线性分类器 (SVM RBF) 评估 latent 表征的分类能力
    """
    if label_key not in adata.obs:
        print(f"'{label_key}' 不在 adata.obs 中。可用标签有：{adata.obs.columns.tolist()}")
        return None

    labels = adata.obs[label_key].astype(str).values
    X_train, X_test, y_train, y_test = train_test_split(latent, labels, test_size=0.2, random_state=42)

    # 非线性分类器 - RBF SVM
    clf = SVC(kernel="rbf", C=10, gamma="scale")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"[非线性分类器 SVM-RBF] 使用 latent 表征预测 '{label_key}' 的准确率: {acc:.4f}")
    return acc

def inter_intra_class_distance_ratio(adata, label_key, latent):
    from itertools import combinations
    if label_key not in adata.obs:
        print(f"'{label_key}' 不在 adata.obs 中。可用标签有：{adata.obs.columns.tolist()}")
        return None

    labels = adata.obs[label_key].astype(str).values
    unique_labels = np.unique(labels)

    # 计算每一类的中心
    class_means = {}
    intra_distances = []

    for lab in unique_labels:
        z_c = latent[labels == lab]
        mu_c = z_c.mean(axis=0)
        class_means[lab] = mu_c

        # 类内距离
        intra = np.mean(np.sum((z_c - mu_c) ** 2, axis=1))
        intra_distances.append(intra)

    D_intra = np.mean(intra_distances)

    # 类间距离
    inter_distances = []
    for lab1, lab2 in combinations(unique_labels, 2):
        mu1 = class_means[lab1]
        mu2 = class_means[lab2]
        inter = np.sum((mu1 - mu2) ** 2)
        inter_distances.append(inter)

    D_inter = np.mean(inter_distances)

    ratio = D_inter / (D_intra + 1e-8)

    print(f"类内距离: {D_intra:.4f}")
    print(f"类间距离: {D_inter:.4f}")
    print(f"类间 / 类内 距离比: {ratio:.4f}")

    return {
        "intra_class_distance": D_intra,
        "inter_class_distance": D_inter,
        "ratio": ratio
    }
    
def get_neighbors_from_adata(adata):
    """从 adata.obsp['distances'] 提取近邻索引"""
    dists = adata.obsp["distances"]
    if not sp.isspmatrix_csr(dists):
        dists = sp.csr_matrix(dists)

    n_neighbors = adata.uns["neighbors"]["params"]["n_neighbors"]

    indices = []
    for i in range(dists.shape[0]):
        row = dists[i].toarray().ravel()
        # argsort 取前 n_neighbors+1（包含自己）
        neighbors = np.argsort(row)[:n_neighbors+1]
        indices.append(neighbors)
    return np.array(indices)

def evaluate_ICCoh(adata, cluster_key="clusters", velocity_key="velocity"):
    """
    计算 Inter-cluster coherence (ICCoh)
    
    参数
    ----
    adata : AnnData
        包含 embedding 和速度的 AnnData 对象
    cluster_key : str
        obs 中的聚类标签键
    velocity_key : str
        adata.layers 中保存速度的键
    
    返回
    ----
    iccoh_score : float
        ICCoh 分数 (0~1)
    """
    import itertools

    print("计算 ICCoh")
    if velocity_key not in adata.layers:
        raise ValueError(f"adata.layers 中未找到 {velocity_key}")
    if cluster_key not in adata.obs:
        raise ValueError(f"adata.obs 中未找到 {cluster_key}")

    V = adata.layers[velocity_key]  # (n_cells, n_features)
    if isinstance(V, np.ndarray):
        velocities = V
    else:
        velocities = V.toarray() if sp.issparse(V) else np.array(V)

    labels = adata.obs[cluster_key].astype("category").cat.codes.values
    unique_labels = np.unique(labels)
    
    cluster_scores = []
    cluster_sizes = []

    for ul in unique_labels:
        idx = np.where(labels == ul)[0]
        if len(idx) < 2:
            continue  # 至少两个细胞才能计算余弦相似度

        v_cluster = velocities[idx]
        # 单位化速度向量
        norms = np.linalg.norm(v_cluster, axis=1, keepdims=True)
        norms[norms < 1e-9] = 1e-9
        v_unit = v_cluster / norms

        # 两两余弦相似度平均
        sims = []
        for i, j in itertools.combinations(range(len(idx)), 2):
            cos_sim = np.dot(v_unit[i], v_unit[j])
            sims.append(cos_sim)
        cluster_score = np.mean(sims)
        cluster_scores.append(cluster_score)
        cluster_sizes.append(len(idx))

    if len(cluster_scores) == 0:
        print("没有可计算 ICCoh 的簇")
        return 0.0

    # 加权平均
    cluster_sizes = np.array(cluster_sizes)
    iccoh_score = np.average(cluster_scores, weights=cluster_sizes)
    print(f"ICCoh score: {iccoh_score:.4f}")
    return iccoh_score

def evaluate_ICCoh_local(adata, cluster_key="clusters", velocity_key="velocity", n_neighbors=None):
    """
    基于局部近邻（kNN内同簇细胞）计算簇内速度方向一致性（Local Inter-cluster Coherence）
    """
    print("计算 Local ICCoh（基于近邻）")


    if velocity_key not in adata.layers:
        raise ValueError(f"adata.layers 中未找到 {velocity_key}")
    if cluster_key not in adata.obs:
        raise ValueError(f"adata.obs 中未找到 {cluster_key}")


    # 获取速度（投影到 embedding 空间更合理）
    V = adata.layers[velocity_key]
    velocities = V.toarray() if sp.issparse(V) else V.copy()
    
    labels = adata.obs[cluster_key].values
    unique_labels = np.unique(labels)


    # 获取邻居索引
    indices = get_neighbors_from_adata(adata)  # shape: (n_cells, n_neighbors+1)
    if n_neighbors is None:
        n_neighbors = indices.shape[1] - 1  # 扣掉自己


    local_sims = []


    for i in range(adata.n_obs):
        ci = labels[i]
        neighbor_idx = indices[i][1:]  # k个近邻（不含自身）
        
        # 筛选同簇邻居
        same_cluster_mask = (labels[neighbor_idx] == ci)
        neighbor_in_cluster = neighbor_idx[same_cluster_mask]
        
        if len(neighbor_in_cluster) == 0:
            continue  # 没有同簇邻居，跳过


        vi = velocities[i]
        vj_batch = velocities[neighbor_in_cluster]


        # 单位化
        vi_norm = np.linalg.norm(vi)
        vj_norms = np.linalg.norm(vj_batch, axis=1)


        if vi_norm < 1e-9 or np.any(vj_norms < 1e-9):
            continue


        vi_unit = vi / vi_norm
        vj_unit = vj_batch / vj_norms[:, None]


        # 计算当前细胞与每个同簇邻居的余弦相似度
        sims = np.dot(vj_unit, vi_unit)
        local_sims.append(np.mean(sims))


    if len(local_sims) == 0:
        print("没有足够的局部同簇邻居用于计算 Local ICCoh")
        return 0.0


    local_iccoh = np.mean(local_sims)
    print(f"Local ICCoh score: {local_iccoh:.4f} (基于 {len(local_sims)} 个细胞的局部邻居)")
    return local_iccoh


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import issparse

def evaluate_CBDir2(adata, cluster_key, velocity_key, cluster_edges, return_raw=False, x_emb="X_umap"):
    def keep_type(adata, nodes, target, cluster_key):
        nodes = np.asarray(nodes)
        nodes = nodes[nodes >= 0]
        return nodes[adata.obs[cluster_key].iloc[nodes].values == target]
    
    scores = {}
    all_scores = {}

    if x_emb == "X_umap":
        v_emb = adata.obsm['{}_umap'.format(velocity_key)]
    else:
        v_emb = adata.obsm[[key for key in adata.obsm if key.startswith(velocity_key)][0]]

    x_emb = adata.obsm[x_emb]

    for u, v in cluster_edges:
        sel = adata.obs[cluster_key] == u
        nbs = adata.uns['neighbors']['indices'][sel]

        boundary_nodes = map(lambda nodes: keep_type(adata, nodes, v, cluster_key), nbs)
        x_points = x_emb[sel]
        x_velocities = v_emb[sel]

        type_score = []
        valid_pairs = 0

        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if len(nodes) == 0:
                continue

            position_dif = x_emb[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1, -1)).flatten()
            type_score.append(np.mean(dir_scores))
            valid_pairs += len(nodes)

        scores[(u, v)] = np.mean(type_score)
        all_scores[(u, v)] = type_score

        print(f"CBDir {u} -> {v}: 有效细胞 = {len(type_score)}, 有效 cell-pair = {valid_pairs}")

    if return_raw:
        return all_scores

    return scores, np.mean([sc for sc in scores.values() if not np.isnan(sc)])

def evaluate_CBDir3(adata, cluster_key, velocity_key, cluster_edges, return_raw=False, x_emb="X_umap"):
    def keep_type(adata, nodes, target, cluster_key):
        """只保留属于目标 cluster 的邻居节点"""
        nodes = np.asarray(nodes)
        nodes = nodes[nodes >= 0]
        return nodes[adata.obs[cluster_key].iloc[nodes].values == target]

    scores = {}
    all_scores = {}

    # 获取速度嵌入
    if x_emb == "X_umap":
        v_emb = adata.obsm['{}_umap'.format(velocity_key)]
    else:
        v_emb = adata.obsm[[key for key in adata.obsm if key.startswith(velocity_key)][0]]

    # 获取坐标嵌入
    x_emb = adata.obsm[x_emb]

    # 获取 velocity graph
    vg = adata.uns['velocity_graph']  # csr_matrix
    n_cells = adata.n_obs

    for u, v in cluster_edges:
        # 布尔索引转 NumPy array
        sel = np.array(adata.obs[cluster_key] == u)

        # 选出对应行
        sel_rows = vg[sel]  # csr_matrix of selected cells

        # 每行非零列作为邻居
        nbs_list = [row.indices for row in sel_rows]

        # boundary_nodes：只保留属于目标 cluster v 的邻居
        boundary_nodes = map(lambda nodes: keep_type(adata, nodes, v, cluster_key), nbs_list)

        x_points = x_emb[sel]
        x_velocities = v_emb[sel]

        type_score = []
        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if len(nodes) == 0:
                continue

            position_dif = x_emb[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1, -1)).flatten()
            type_score.append(np.mean(dir_scores))

        # 平均方向一致性分数
        scores[(u, v)] = np.mean(type_score) if len(type_score) > 0 else np.nan
        all_scores[(u, v)] = type_score

    if return_raw:
        return all_scores

    # 返回各边分数 + 所有非 NaN 平均
    avg_score = np.mean([sc for sc in scores.values() if not np.isnan(sc)])
    return scores, avg_score


def compute_velocity_from_graph(
    adata,
    embedding="umap",
    new_key="velocity"
):
    """
    根据 velocity_graph 在低维嵌入空间中计算速度向量

    Parameters
    ----------
    adata : AnnData
        包含 adata.uns['velocity_graph'] 和 adata.obsm['X_{embedding}']
    embedding : str
        低维嵌入名称（如 'umap'）
    new_key : str
        结果速度向量存储到 adata.obsm[new_key]

    Returns
    -------
    adata : AnnData
    """

    if 'velocity_graph' not in adata.uns:
        raise ValueError("adata.uns 中未找到 'velocity_graph'")

    embedding_key = f"X_{embedding}"
    if embedding_key not in adata.obsm:
        raise ValueError(f"adata.obsm 中未找到 '{embedding_key}'")

    vg = adata.uns['velocity_graph']          # sparse matrix (n_cells, n_cells)
    x_emb = adata.obsm[embedding_key]         # (n_cells, dim)

    n_cells, dim = x_emb.shape
    velocities = np.zeros((n_cells, dim))

    for i in range(n_cells):
        row = vg[i]
        neighbors = row.indices
        weights = row.data

        if len(neighbors) == 0 or weights.sum() == 0:
            continue

        directions = x_emb[neighbors] - x_emb[i]
        velocities[i] = (directions * weights[:, None]).sum(axis=0) / weights.sum()

    adata.obsm[new_key] = velocities
    return adata