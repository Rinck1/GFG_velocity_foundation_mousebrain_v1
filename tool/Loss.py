import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def tangent_alignment_loss(E_m, E_v):
    E_m = F.normalize(E_m, dim=1)
    E_v = F.normalize(E_v, dim=1)
    dot = F.cosine_similarity(E_m, E_v, dim=1)
    return (dot ** 2).mean()

def entropy_uniform_loss(probs, eps=1e-6):
    mean_probs_per_gene = probs.mean(dim=0)  # (G, K)
    uniform = torch.full_like(mean_probs_per_gene, 1.0 / mean_probs_per_gene.size(-1))
    kl = F.kl_div((mean_probs_per_gene + eps).log(), uniform, reduction='none').sum(dim=1)
    return kl.mean()

def balanced_entropy_loss(probs, eps=1e-6):
    mean_probs_per_gene = probs.mean(dim=0)
    entropy = -(mean_probs_per_gene * (mean_probs_per_gene + eps).log()).sum(dim=1).mean()
    per_sample_entropy = -(probs * (probs + eps).log()).sum(dim=-1).mean()
    return (entropy - per_sample_entropy) ** 2

def compute_loss_pred(pred, target):
    return F.mse_loss(pred, target)

def compute_loss_smooth(v_pred, adj, eps=1e-8):
    if v_pred.dim() == 3:
        v = v_pred.squeeze(-1)
    else:
        v = v_pred
    v_norm = v / (v.norm(dim=1, keepdim=True) + eps)  # (B, G)
    cos_sim = v_norm @ v_norm.T  # 矩阵乘法代替广播
    loss = (adj * (1 - cos_sim)).sum() / (adj.sum() + eps)
    return loss

def compute_loss_align(v_pred, x_true, adj, abs=True, eps=1e-10):
    """
    v_pred: (B, G)
    x_true: (B, G)
    adj: (B, B)
    
    对每条边(i,j)，计算余弦相似：
        cos( v_pred[i],  s_true[j] - s_true[i] )
        
    改变：梯度截断
    """
    # 获取邻居对 (i, j)
    i, j = adj.nonzero(as_tuple=True)
    if i.numel() == 0:
        return v_pred.sum() * 0.0
    # 取节点 i, j 的向量
    v_i = v_pred[i]           # (E, G)
    ds  = x_true[j] - x_true[i]  # (E, G)
    
    # 余弦相似度
    cosi = F.cosine_similarity(v_i, ds, dim=1, eps=eps)  # (E,)
    
    # loss = (1 - cosi.abs())**3 + (1 - cosj.abs())**3
    if abs:
        loss = 1 - cosi.abs()
    else:
        loss = 1 - cosi
    edge_weights = adj[i, j].to(dtype=loss.dtype)
    return (loss * edge_weights).sum() / (edge_weights.sum() + eps)

def compute_loss_align_pca(v_pred, x_true, adj, abs=False, pcs=2, eps=1e-10):
    """
    v_pred: (B, G)
    x_true: (B, G)
    adj: (B, B)

    高维天然就是正交的(?)，PCA后才能收敛
    """

    # ====== 用 x_true 的低秩 PCA 获取投影矩阵 ======
    # x_true: (B, G)
    U, S, V = torch.pca_lowrank(x_true, q=pcs)   # V: (G, 30)

    # ====== 投影到 30 维 ======
    # x_pca = x * V
    x_pca = x_true @ V                          # (B, 30)
    v_pca = v_pred @ V                          # (B, 30)

    # ====== 获取邻居对 (i, j) ======
    i, j = adj.nonzero(as_tuple=True)
    if i.numel() == 0:
        return v_pred.sum() * 0.0

    v_i = v_pca[i]              # (E, 30)
    ds  = x_pca[j] - x_pca[i]   # (E, 30)

    # ====== 计算余弦对齐 loss ======
    cosi = F.cosine_similarity(v_i, ds, dim=1, eps=eps)
    
    if abs:
        loss = 1 - cosi.abs()
    else:
        loss = 1 - cosi
    
    edge_weights = adj[i, j].to(dtype=loss.dtype)
    return (loss * edge_weights).sum() / (edge_weights.sum() + eps)
