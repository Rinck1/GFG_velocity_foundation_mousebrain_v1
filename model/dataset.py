import torch
from torch.utils.data import Dataset
import scipy.sparse
import numpy as np

class Data(Dataset):
    def __init__(self, adata, spliced_key="spliced", unspliced_key="unspliced", root_key='is_root'):
        self.adata = adata
        
        def to_dense(layer):
            if scipy.sparse.issparse(layer):
                return layer.toarray()
            return layer
        
        self.spliced = to_dense(adata.layers[spliced_key])
        self.unspliced = to_dense(adata.layers[unspliced_key])
        
        # 计算归一化状态（保持不变）
        self.spliced_state = {
            "mean": self.spliced.mean(axis=0, keepdims=True).flatten(), 
            "std": (self.spliced.std(axis=0, keepdims=True) + 1e-8).flatten()
        }
        self.unspliced_state = {
            "mean": self.unspliced.mean(axis=0, keepdims=True).flatten(), 
            "std": (self.unspliced.std(axis=0, keepdims=True) + 1e-8).flatten()
        }
        self.layer_states = {"unspliced": self.unspliced_state, "spliced": self.spliced_state}

        # 拼成 (N, 2G): 先 u, 后 s
        self.data = np.concatenate([self.unspliced, self.spliced], axis=1)

        # 新增：读取 root mask（支持自动兜底）
        if root_key in adata.obs:
            self.root_mask = adata.obs[root_key].values.astype(bool)
        else:
            # 自动兜底：用 unspliced 比例最高的前 50 个细胞
            u = self.unspliced.sum(axis=1)
            s = self.spliced.sum(axis=1)
            ratio = u / (u + s + 1e-8)
            top50 = np.argsort(ratio)[-50:]
            self.root_mask = np.zeros(len(adata), dtype=bool)
            self.root_mask[top50] = True
            print("No 'is_root' found, auto using top-50 highest u/(u+s) cells as root")

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        x = self.data[idx]  # (2G,)
        root = self.root_mask[idx]  # bool scalar
        # 返回原始细胞索引，确保 shuffle 后仍能切出与 batch 对齐的邻接子矩阵。
        return torch.tensor(x, dtype=torch.float32), root, idx