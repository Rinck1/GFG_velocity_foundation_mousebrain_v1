import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
from sklearn.decomposition import PCA
import scanpy as sc

def build_neighbor_indices(adata, n_neighbors=30):
    """
    根据 adata.obsp['distances'] 生成 adata.uns['neighbors']['indices']。
    若 distances 不存在，会自动运行 sc.pp.neighbors。
    """
    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=30)
    
    # 确保 distances 矩阵存在
    if "distances" not in adata.obsp:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=30)
    
    # 将距离矩阵转为普通数组
    dist = adata.obsp["distances"].toarray()
    # 对每一行找到距离最近的 n_neighbors 个点（不包括自己）
    indices = np.argsort(dist, axis=1)[:, 1:n_neighbors+1]
    
    adata.uns["neighbors"]["indices"] = indices
    print(f"[build_neighbor_indices] 已生成邻居索引，形状为 {indices.shape}")
    return adata

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
def evaluate_CBDir(adata, cluster_key="clusters", velocity_key="velocity", cluster_edges=None):
    print("计算 CBDir")

    if velocity_key not in adata.layers:
        raise ValueError(f"adata.layers 中未找到 {velocity_key}")
    if cluster_key not in adata.obs:
        raise ValueError(f"adata.obs 中未找到 {cluster_key}")
    if "X_umap" not in adata.obsm:
        raise ValueError("需要在 adata.obsm['X_umap'] 中提供低维嵌入 (例如 UMAP)")

    # 坐标 (embedding)，通常是 UMAP
    X = adata.obsm["X_umap"]
    # 速度
    V = adata.layers[velocity_key]
    # 过滤 NaN 值
    valid_cells = ~np.isnan(V).any(axis=1)
    if not valid_cells.any():
        print("所有细胞的速度向量均包含 NaN，无法计算 CBDir")
        return 0.0

    X = X[valid_cells]
    V = V[valid_cells]
    labels = adata.obs[cluster_key][valid_cells].astype("category")
    label_codes = labels.cat.codes.values
    label_names = labels.cat.categories
    adata_valid = adata[valid_cells].copy()

    # 如果速度在基因空间，投影到 embedding 空间
    if V.shape[1] != X.shape[1]:
        pca = PCA(n_components=X.shape[1])
        V = pca.fit_transform(V)

    # 获取邻居索引（需要用户定义的函数）
    indices = get_neighbors_from_adata(adata_valid)
    n = X.shape[0]

    # 若给定 cluster_edges，则建立映射关系，方便比较
    allowed_edges = None
    if cluster_edges is not None:
        allowed_edges = set(cluster_edges)

    cbdir_scores = []
    for i in range(n):
        xi = X[i]
        vi = V[i]
        ci_name = labels.iloc[i]

        neighbor_idx = indices[i][1:]  # 去掉自己
        cross_boundary = []

        for j in neighbor_idx:
            cj_name = labels.iloc[j]
            if ci_name != cj_name:
                # 如果指定了 cluster_edges，则只计算允许的方向
                if allowed_edges is not None:
                    if (ci_name, cj_name) not in allowed_edges:
                        continue
                cross_boundary.append(j)

        if len(cross_boundary) == 0:
            continue

        cos_sims = []
        for j in cross_boundary:
            direction = X[j] - xi
            if np.linalg.norm(direction) < 1e-9 or np.linalg.norm(vi) < 1e-9:
                continue
            cos_sim = np.dot(vi, direction) / (
                np.linalg.norm(vi) * np.linalg.norm(direction)
            )
            if not np.isnan(cos_sim):
                cos_sims.append(cos_sim)

        if len(cos_sims) > 0:
            cbdir_scores.append(np.mean(cos_sims))

    cbdir_score = np.mean(cbdir_scores) if len(cbdir_scores) > 0 else 0.0
    print(f"CBDir score: {cbdir_score:.4f} (基于 {len(cbdir_scores)} 个细胞)")
    return cbdir_score

def evaluate_CBDir2(adata, cluster_key, velocity_key, cluster_edges, return_raw=False, x_emb="X_umap", n_neighbor=200):
    def keep_type(adata, nodes, target, cluster_key):
        return nodes[adata.obs[cluster_key].iloc[nodes].values == target]

    scores, all_scores = {}, {}

    if x_emb == "X_umap":
        v_emb = adata.obsm[f"{velocity_key}_umap"]
    else:
        v_emb = adata.obsm[[key for key in adata.obsm if key.startswith(velocity_key)][0]]
    x_emb = adata.obsm[x_emb]

    if n_neighbor is None:
        neighbors = adata.uns["neighbors"]["indices"]
    else:
        from sklearn.neighbors import NearestNeighbors
        neighbors = NearestNeighbors(n_neighbors=n_neighbor + 1).fit(x_emb).kneighbors(x_emb, return_distance=False)[:, 1:]

    for u, v in cluster_edges:
        sel = adata.obs[cluster_key].values == u
        nbs = neighbors[sel]
        boundary_nodes = map(lambda nodes: keep_type(adata, nodes, v, cluster_key), nbs)
        x_points, x_velocities = x_emb[sel], v_emb[sel]
        type_score, valid_pairs = [], 0

        for x_pos, x_vel, nodes in zip(x_points, x_velocities, boundary_nodes):
            if len(nodes) == 0:
                continue
            position_dif = x_emb[nodes] - x_pos
            dir_scores = cosine_similarity(position_dif, x_vel.reshape(1, -1)).flatten()
            type_score.append(np.mean(dir_scores))
            valid_pairs += len(nodes)

        scores[(u, v)] = np.mean(type_score) if type_score else np.nan
        all_scores[(u, v)] = type_score
        total_cells = np.sum(sel)
        print(f"CBDir {u} -> {v}: 有效细胞 = {len(type_score)}/{total_cells} ({len(type_score)/total_cells:.2%}), 有效 cell-pair = {valid_pairs}")

    if return_raw:
        return all_scores
    valid_scores = [sc for sc in scores.values() if not np.isnan(sc)]
    return scores, np.mean(valid_scores) if valid_scores else np.nan

def evaluate_ICCoh(adata, cluster_key="clusters", velocity_key="velocity"):
    """
    计算 Inter-cluster coherence (ICCoh)，处理 NaN 值
    """
    import itertools

    print("计算 ICCoh")
    if velocity_key not in adata.layers:
        raise ValueError(f"adata.layers 中未找到 {velocity_key}")
    if cluster_key not in adata.obs:
        raise ValueError(f"adata.obs 中未找到 {cluster_key}")

    V = adata.layers[velocity_key]
    # 过滤 NaN 值
    valid_cells = ~np.isnan(V).any(axis=1)
    if not valid_cells.any():
        print("所有细胞的速度向量均包含 NaN，无法计算 ICCoh")
        return 0.0

    V = V[valid_cells]
    labels = adata.obs[cluster_key][valid_cells].astype("category").cat.codes.values
    adata_valid = adata[valid_cells].copy()

    if sp.issparse(V):
        velocities = V.toarray()
    else:
        velocities = np.array(V)

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
        valid_idx = norms.ravel() > 1e-9  # 过滤零向量
        if valid_idx.sum() < 2:
            continue  # 有效细胞不足

        v_cluster = v_cluster[valid_idx]
        norms = norms[valid_idx]
        v_unit = v_cluster / norms

        # 两两余弦相似度平均
        sims = []
        for i, j in itertools.combinations(range(len(v_cluster)), 2):
            cos_sim = np.dot(v_unit[i], v_unit[j])
            if not np.isnan(cos_sim):  # 确保余弦相似度有效
                sims.append(cos_sim)
        if len(sims) > 0:
            cluster_score = np.mean(sims)
            cluster_scores.append(cluster_score)
            cluster_sizes.append(len(v_cluster))

    if len(cluster_scores) == 0:
        print("没有可计算 ICCoh 的簇")
        return 0.0

    # 加权平均
    cluster_sizes = np.array(cluster_sizes)
    iccoh_score = np.average(cluster_scores, weights=cluster_sizes)
    print(f"ICCoh score: {iccoh_score:.4f}")
    return iccoh_score

def evaluate_local_ICCoh(adata, cluster_key="clusters", velocity_key="velocity", return_raw=False):

    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    if 'neighbors' not in adata.uns or 'indices' not in adata.uns['neighbors']:
        raise ValueError("请先在 adata.uns['neighbors'] 中计算邻居信息！")

    if velocity_key not in adata.layers:
        raise ValueError(f"adata.layers 中未找到 '{velocity_key}'")

    if cluster_key not in adata.obs:
        raise ValueError(f"adata.obs 中未找到 '{cluster_key}'")

    velocities = adata.layers[velocity_key]
    if sp.issparse(velocities):
        velocities = velocities.toarray()
    velocities = np.array(velocities)

    # 去除 NaN 或零向量
    norms = np.linalg.norm(velocities, axis=1)
    valid_cells = (~np.isnan(velocities).any(axis=1)) & (norms > 1e-9)
    if not np.any(valid_cells):
        print("所有细胞的速度向量均无效（NaN 或零向量）")
        return {}, 0.0

    adata = adata[valid_cells].copy()
    velocities = velocities[valid_cells]
    labels = adata.obs[cluster_key].astype("category").values
    nbs = adata.uns['neighbors']['indices'][valid_cells]

    clusters = np.unique(labels)
    scores = {}
    all_scores = {}

    for cat in clusters:
        sel = labels == cat
        if sel.sum() < 2:
            continue

        cat_vels = velocities[sel]
        cat_nbs = nbs[sel]

        # 过滤出同类邻居
        cat_nb_nodes = []
        for nodes in cat_nbs:
            same_cat_nodes = np.where(labels[nodes] == cat)[0]
            cat_nb_nodes.append(same_cat_nodes)

        # 计算邻居间余弦相似度
        cat_score_list = []
        for ith, nodes in enumerate(cat_nb_nodes):
            if len(nodes) == 0:
                continue
            try:
                sim = cosine_similarity(cat_vels[[ith]], velocities[nodes]).mean()
                if not np.isnan(sim):
                    cat_score_list.append(sim)
            except Exception:
                continue

        if len(cat_score_list) > 0:
            scores[cat] = np.mean(cat_score_list)
            all_scores[cat] = cat_score_list

    if len(scores) == 0:
        print("没有可计算 local ICCoh 的簇")
        return {}, 0.0

    mean_score = np.mean([v for v in scores.values() if not np.isnan(v)])

    if return_raw:
        return all_scores

    print(f"Local ICCoh mean score: {mean_score:.4f}")
    return scores, mean_score