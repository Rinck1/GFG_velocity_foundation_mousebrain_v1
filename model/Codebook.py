import torch
import torch.nn as nn
import torch.nn.functional as F
    
    
class SoftVectorQuantizer(nn.Module):
    """
    简化版 Soft VQ:
      - 用 softmax(-dist / tau) 得到软分配
      - 码本可训练 (nn.Parameter)
      - 保留 commit loss + 熵正则
    """
    def __init__(self, num_codes: int, code_dim: int,
                 beta: float = 1.0, gamma: float = 1.0,
                 init_scale: float = 1.0, tau: float = 1.0,
                 metric: str = "euclidean", normalize: bool = False):
        super().__init__()
        self.K = num_codes
        self.D = code_dim
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.metric = metric
        self.normalize = normalize

        embed = torch.randn(self.K, self.D) * init_scale
        self.embedding = nn.Parameter(embed)  # (K, D)

    def _perplexity(self, probs):
        # probs: (N, K)
        avg_probs = probs.mean(dim=0) + 1e-10
        entropy = -(avg_probs * avg_probs.log()).sum()
        return torch.exp(entropy)

    def forward(self, z):
        """
        z : (B, G, D) 共享码本
        """
        B, G, D = z.shape
        assert D == self.D

        z_flat = z.view(-1, D)   # (N, D), N = B*G
        E = self.embedding       # (K, D)

        # -------- 距离计算 --------
        if self.metric == "cosine":
            z_n = F.normalize(z_flat, dim=-1)
            e_n = F.normalize(E, dim=-1)
            logits = z_n @ e_n.t()   # (N, K)
            probs = F.softmax(logits / self.tau, dim=-1)
        else:  # euclidean
            z_sq = (z_flat ** 2).sum(dim=1, keepdim=True)  # (N, 1)
            e_sq = (E ** 2).sum(dim=1, keepdim=True).t()   # (1, K)
            dist = z_sq + e_sq - 2 * (z_flat @ E.t())      # (N, K)
            probs = F.softmax(-dist / self.tau, dim=-1)

        # -------- 量化向量 --------
        z_q_flat = probs @ E   # (N, D)
        z_q = z_q_flat.view(B, G, D)

        # -------- 损失 --------
        commitment_loss = F.mse_loss(z_flat, z_q_flat.detach())
        # 熵正则
        avg_probs = probs.mean(dim=0)  # (K,)
        entropy = -(avg_probs * (avg_probs + 1e-10).log()).sum()
        max_entropy = torch.log(torch.tensor(self.K, dtype=z.dtype, device=z.device))
        entropy_loss = max_entropy - entropy

        vq_loss = self.beta * commitment_loss + self.gamma * entropy_loss

        ppl = self._perplexity(probs)
        loss_dict = {"vq": vq_loss, "commit": commitment_loss, "perplexity": ppl}

        # 取 argmax 索引（仅用于分析，不影响训练）
        codes = torch.argmax(probs, dim=1).view(B, G)

        return z_q, loss_dict, codes
