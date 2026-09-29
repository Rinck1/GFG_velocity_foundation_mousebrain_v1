import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from tool.utils import re_z_score

# ------------------------------
# 1) 状态解码器 g_psi: z_s -> x=(u,s)
# ------------------------------
class BaseDecoder(nn.Module):
    def __init__(self, latent_dim: int, num_genes: int, hidden=(256, 256)):#hidden=(128, 32)):
        super().__init__()
        layers = []
        d = latent_dim
        for h in hidden:
            # layers.append(nn.LayerNorm(d))
            layers.append(nn.Linear(d, h))
            layers.append(nn.GELU())     
            layers.append(nn.Dropout(0.2))
            d = h
        # 输出 2G 维，对应 (u, s)
        layers += [nn.Linear(d, 2)]
        self.net = nn.Sequential(*layers)
        self.num_genes = num_genes
        
    def forward(self, z_s):
        # z_s: (B * G, K)
        x = self.net(z_s)             # (B*G, 2)
        return x

# ----------------------------------------
# 2) NTPL: v_x = J_g(z_s) @ z_v  (JVP实现)
# ----------------------------------------
def ntpl_jvp(decoder: nn.Module, z_s: torch.Tensor, z_v: torch.Tensor):
    """
    decoder: StateDecoder (或任意 g(z))
    z_s: (B, d)
    z_v: (B, d) 作为切向方向的潜在速度
    return:
      v_x: (B, 2G) 观测空间切向速度
    """
    z_s = z_s.requires_grad_(True)
    _, v_x = torch.autograd.functional.jvp(decoder.net, (z_s,), (z_v,), create_graph=True)
    
    return v_x  # (B, 2G)

def check_param(param, name="param"):
    """
    简单检查 alpha, beta 或 gamma 的基本统计信息。
    """
    p = param.ravel()  # 比 flatten() 稍快，不拷贝内存
    n = p.numel()
    if n == 0:
        print(f"{name}: empty")
        return
    print(f"{name}: mean={p.mean():.4f}, min={p.min():.4f}, max={p.max():.4f}")
# ---------------------------------------------------
# 3) ODE 最小残差（不显式学习 alpha,beta,gamma）
# ---------------------------------------------------
def ode_min_residual_loss(u, s, v_u, v_s, lam: float = 1e-3, lam_nonneg: float = 1e-3, 
                          mask: Optional[torch.Tensor] = None, 
                          nonneg: str = "penalty"):
    """
    RNA velocity 动力学模型的最小残差闭式解损失函数。
    --------------------------------------------------------

    参数
    ------
    u : torch.Tensor, 形状 (B, G)
        未剪接 RNA 表达量矩阵。
        B = 细胞数，G = 基因数。
    s : torch.Tensor, 形状 (B, G)
        剪接 RNA 表达量矩阵。
    v_u : torch.Tensor, 形状 (B, G)
        对应的未剪接 RNA 的观测速度 (du/dt)，通常来自估计或模型预测。
    v_s : torch.Tensor, 形状 (B, G)
        对应的剪接 RNA 的观测速度 (ds/dt)。

    lam : float, 默认 1e-3
        岭回归正则项 λ，用于稳定矩阵求解 (防止奇异或过拟合)。

    lam_nonneg : float, 默认 1e-3
        当启用非负约束 (nonneg="penalty") 时，负速率惩罚项的系数。

    mask : torch.Tensor 或 None, 形状 (B, G), 默认 None
        有效数据掩码，用于屏蔽无效细胞或特定基因位置。

    nonneg : str, 默认 "penalty"
        控制速率参数 (α, β, γ) 的非负性约束方式：
            - "none"     : 不约束；
            - "clamp"    : 直接将负值截断为 0；
            - "softplus" : 使用 softplus 激活保证非负；
            - "penalty"  : 对负值加惩罚项。

    返回
    ------
    loss : torch.Tensor
        标量损失值，表示残差平方和 + 负值惩罚项。

    aux : dict
        调试/可视化用中间变量字典，包括：
            - "alpha": 转录速率 (α)
            - "beta":  剪接速率 (β)
            - "gamma": 降解速率 (γ)
            - "Mu": 预测的 du/dt
            - "Ms": 预测的 ds/dt
            - "r2": 每个点的残差平方 (v_u - Mu)^2 + (v_s - Ms)^2
    """
    assert u.shape == s.shape == v_u.shape == v_s.shape
    B, G = u.shape
    device, original_dtype = u.device, u.dtype
    # Keep the tiny kinetic solve in fp32 under mixed precision.  The
    # encoder/decoder can remain BF16, but the 3x3 system is unstable and is
    # unsupported by some CUDA kernels in a reduced precision dtype.
    solve_dtype = (
        torch.float32
        if original_dtype in (torch.float16, torch.bfloat16)
        else original_dtype
    )
    u = u.to(solve_dtype)
    s = s.to(solve_dtype)
    v_u = v_u.to(solve_dtype)
    v_s = v_s.to(solve_dtype)
    dtype = solve_dtype

    # 构造 A = M^T M + lam I, b = M^T v  （向量化写法）
    # M = [[1, -u, 0], [0, u, -s]]
    a11 = torch.ones_like(u)
    a12 = -u
    a13 = torch.zeros_like(u)
    a22 = 2 * (u ** 2)
    a23 = -(u * s)
    a33 = (s ** 2)

    A = torch.stack([
        torch.stack([a11, a12, a13], dim=-1),
        torch.stack([a12, a22, a23], dim=-1),
        torch.stack([a13, a23, a33], dim=-1),
    ], dim=-2)  # (B, G, 3, 3)

    eye = torch.eye(3, device=device, dtype=dtype).view(1, 1, 3, 3)
    A = A + lam * eye

    # b = [v_u, u*(v_s - v_u), -s*v_s]^T
    b1 = v_u
    b2 = u * (v_s - v_u)
    b3 = -s * v_s
    b = torch.stack([b1, b2, b3], dim=-1).unsqueeze(-1)  # (B,G,3,1)

    # 解 theta*（可微分）
    theta = torch.linalg.solve(A, b).squeeze(-1)  # (B,G,3)
    alpha, beta, gamma = theta.unbind(dim=-1)

    if nonneg == "clamp":
        alpha = torch.clamp(alpha, min=0.0)
        beta = torch.clamp(beta, min=0.0)
        gamma = torch.clamp(gamma, min=0.0)
    elif nonneg == "softplus":
        alpha = F.softplus(alpha)
        beta = F.softplus(beta)
        gamma = F.softplus(gamma)
    elif nonneg == "penalty":
        penalty = (F.relu(-alpha) + F.relu(-beta) + F.relu(-gamma)).mean() * lam_nonneg

    # 由 M theta* 得到“投影后”的 ODE 速度
    Mu = alpha - beta * u           # (B,G)
    Ms = beta * u - gamma * s       # (B,G)

    # check_param(alpha, name="alpha")
    # check_param(beta, name="beta")
    # check_param(gamma, name="gamma")
    # 残差
    ru = v_u - Mu
    rs = v_s - Ms
    r2 = ru.pow(2) + rs.pow(2)
    

    if mask is not None:
        mask = mask.to(dtype=r2.dtype)
        r2 = r2 * mask

    if nonneg == "penalty":
        if mask is None:
            residual = r2.mean()
            valid = r2.new_tensor(1.0)
        else:
            valid = mask.sum().clamp_min(1.0)
            residual = r2.sum() / valid
        loss = residual + penalty
    else:
        if mask is None:
            loss = r2.mean()
        else:
            loss = r2.sum() / mask.sum().clamp_min(1.0)
    aux = {"alpha": alpha, "beta": beta, "gamma": gamma, "Mu": Mu, "Ms": Ms, "r2": r2}
    return loss, aux


def ode_shared_gene_residual_loss(
    u, s, v_u, v_s, lam: float = 1e-3, lam_nonneg: float = 1e-3,
    mask: Optional[torch.Tensor] = None, nonneg: str = "none",
):
    """RNA ODE residual with one rate triplet shared across cells per gene.

    The previous ``ode_min_residual_loss`` solves three rates independently
    for every cell-by-gene location although that location supplies only two
    equations.  Its residual can therefore collapse to almost zero without
    learning a cross-cell dynamical law.  This variant stacks all valid cells
    for each gene into one ridge solve, yielding ``(alpha,beta,gamma)`` of
    shape ``(G,3)`` and a meaningful held-out residual.
    """
    if not (u.shape == s.shape == v_u.shape == v_s.shape):
        raise ValueError("u, s, v_u and v_s must have the same shape")
    if u.ndim != 2:
        raise ValueError("shared gene ODE expects tensors shaped (B,G)")
    solve_dtype = torch.float32 if u.dtype in (torch.float16, torch.bfloat16) else u.dtype
    u, s, v_u, v_s = (x.to(solve_dtype) for x in (u, s, v_u, v_s))
    B, G = u.shape
    valid = torch.ones_like(u, dtype=solve_dtype) if mask is None else mask.to(solve_dtype)
    # Invalid entries are excluded from every sufficient statistic.
    uu = u * valid; ss = s * valid
    vu = v_u * valid; vs = v_s * valid
    n = valid.sum(0)
    su = uu.sum(0); ssum = ss.sum(0)
    suu = (uu * u).sum(0); sus = (uu * s).sum(0); sss = (ss * s).sum(0)
    A = torch.stack((
        torch.stack((n, -su, torch.zeros_like(n)), -1),
        torch.stack((-su, 2.0 * suu, -sus), -1),
        torch.stack((torch.zeros_like(n), -sus, sss), -1),
    ), -2)
    eye = torch.eye(3, device=u.device, dtype=solve_dtype).unsqueeze(0)
    A = A + lam * eye
    b = torch.stack((vu.sum(0), (uu * (vs - vu)).sum(0), -(ss * vs).sum(0)), -1)
    theta = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
    alpha, beta, gamma = theta.unbind(-1)
    if nonneg == "clamp":
        alpha, beta, gamma = alpha.clamp_min(0), beta.clamp_min(0), gamma.clamp_min(0)
        penalty = theta.new_zeros(())
    elif nonneg == "softplus":
        alpha, beta, gamma = F.softplus(alpha), F.softplus(beta), F.softplus(gamma)
        penalty = theta.new_zeros(())
    elif nonneg == "penalty":
        penalty = lam_nonneg * (
            F.relu(-alpha).mean() + F.relu(-beta).mean() + F.relu(-gamma).mean()
        )
    elif nonneg == "none":
        penalty = theta.new_zeros(())
    else:
        raise ValueError(f"unknown nonneg mode: {nonneg}")
    mu_u = alpha.unsqueeze(0) - beta.unsqueeze(0) * u
    mu_s = beta.unsqueeze(0) * u - gamma.unsqueeze(0) * s
    r2 = ((v_u - mu_u).square() + (v_s - mu_s).square()) * valid
    loss = r2.sum() / valid.sum().clamp_min(1.0) + penalty
    return loss, {
        "alpha": alpha, "beta": beta, "gamma": gamma,
        "Mu": mu_u, "Ms": mu_s, "r2": r2,
        "valid_fraction": valid.mean().detach(),
    }

def split_us(x, G):
    ''' x: (B*G, 2) '''
    B = x.shape[0] // G
    x = x.view(B, G, 2)
    u = x[:, :, 0]       # (B, G)
    s = x[:, :, 1]       # (B, G)

    return u, s
# ---------------------------------------------------
# 4) 整合模块：解码器 + NTPL + ODE residual（一步到位）
# ---------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, latent_dim: int, num_genes: int, hidden=(256, 256)):
        super().__init__()
        self.statedecoder = BaseDecoder(latent_dim, num_genes, hidden)
        # self.velocitydecoder = BaseDecoder(latent_dim, num_genes, hidden)
        self.num_genes = num_genes

    def forward(self, z_s, z_v, x_obs=None,
                use_recon_for_ode: bool = False,
                lam: float = 1e-3,
                mask: Optional[torch.Tensor] = None, 
                nonneg: str = "penalty", layer_states=None):
        """
        输入:
        - z_s: 状态潜变量 (B * G, K)
        - z_v: 速度潜变量 (B * G, K) 作为 NTPL 的切向方向
        - x_obs: (可选) 观测 (u,s) 拼接，若提供可用于ODE约束（更稳）
        输出:
        - x_hat: 重构 (u_hat, s_hat) 拼接
        - v_x:   观测空间速度 (v_u, v_s) 拼接
        - losses: dict 包含 L_ode 及中间量
        """
        # 1) 解码状态
        x_hat = self.statedecoder(z_s)  # (B,2G),(B,G),(B,G)
        u_hat, s_hat = split_us(x_hat, self.num_genes)
        if layer_states is not None:
            u_hat = re_z_score(u_hat, layer_states["unspliced"]["mean"], layer_states["unspliced"]["std"])
            s_hat = re_z_score(s_hat, layer_states["spliced"]["mean"], layer_states["spliced"]["std"])
        x_hat = torch.cat([u_hat, s_hat], dim=1)
        
        # v_x, _, _ = self.velocitydecoder(z_v, layer_states=layer_states)
        # 2) NTPL 计算观测空间速度
        v_x = ntpl_jvp(self.statedecoder, z_s, z_v)   # (B*G,2)
        # v_x = v_x / (v_x.abs().max(dim=1, keepdim=True).values + 1e-6)  # 归一化
        v_u, v_s = split_us(v_x, self.num_genes)
        if layer_states is not None:
            v_u = re_z_score(v_u, layer_states["unspliced"]["mean"] * 0, layer_states["unspliced"]["std"])
            v_s = re_z_score(v_s, layer_states["spliced"]["mean"] * 0, layer_states["spliced"]["std"])
        v_x = torch.cat([v_u, v_s], dim=1)

        # 3) 选用 ODE 残差中的 (u,s) 来源：观测 or 重构
        if (x_obs is not None) and (not use_recon_for_ode):
            u, s = torch.chunk(x_obs, 2, dim=-1)
        else:
            u, s = u_hat, s_hat

        # 4) 计算 ODE 最小残差
        L_ode, aux = ode_min_residual_loss(u, s, v_u, v_s, lam=lam, mask=mask, nonneg=nonneg)

        losses = {"L_ode": L_ode, **aux}
        
        # print(f"v_x min = {v_x.min().item():.4f}, max = {v_x.max().item():.4f}")
        return x_hat, v_x, losses
