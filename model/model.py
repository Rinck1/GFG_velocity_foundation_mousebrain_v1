import torch
import torch.nn as nn
import torch.nn.functional as F
import math, os
import numpy as np

from model.Encoder import Encoder
from model.Decoder import Decoder
from model.Codebook import SoftVectorQuantizer
from tool.Loss import *
from torch.utils.data import DataLoader
from model.Config import Config


def slice_batch_adjacency(adjacency_matrix, batch_indices, device="cpu"):
    """Return the adjacency submatrix for the exact cells in a shuffled batch."""
    shape = getattr(adjacency_matrix, "shape", None)
    if shape is None or len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"adjacency_matrix must be square, got shape={shape}")

    indices = torch.as_tensor(batch_indices, dtype=torch.long).detach().cpu()
    if indices.ndim != 1:
        raise ValueError(f"batch_indices must be one-dimensional, got shape={tuple(indices.shape)}")
    if indices.numel() and (indices.min().item() < 0 or indices.max().item() >= shape[0]):
        raise IndexError(f"batch index is outside adjacency_matrix with shape={shape}")

    if torch.is_tensor(adjacency_matrix):
        source_indices = indices.to(adjacency_matrix.device)
        batch_adj = adjacency_matrix.index_select(0, source_indices).index_select(1, source_indices)
        return batch_adj.to(device=device, dtype=torch.float32)

    numpy_indices = indices.numpy()
    batch_adj = adjacency_matrix[numpy_indices][:, numpy_indices]
    if hasattr(batch_adj, "toarray"):
        batch_adj = batch_adj.toarray()
    return torch.as_tensor(batch_adj, dtype=torch.float32, device=device)


class VeloModel(nn.Module):
    def __init__(self, dataloader: DataLoader, config : Config, device="cpu", cluster_edge=None):
        super(VeloModel, self).__init__()
        self.G = config.config["data"]["num_gene"]
        self.K = config.config["model"]["codebook_size"]
        self.gene_dim = config.config["model"]["gene_dim"]
        self.use_vq = config.config["model"].get("use_vq", True)
        hidden_state = config.config["model"]["hidden_state"]
        encoder_kwargs = {
            "attention_heads": config.config["model"].get("attention_heads", 4),
            "attention_layers": config.config["model"].get("attention_layers", 1),
            "attention_ff_mult": config.config["model"].get("attention_ff_mult", 2),
            "mlp_dropout": config.config["model"].get("mlp_dropout", 0.2),
            "attention_dropout": config.config["model"].get("attention_dropout", 0.1),
        }
        self.manifold_encoder = Encoder(
            num_genes=self.G,
            gene_dim=self.gene_dim,
            hidden=hidden_state,
            **encoder_kwargs,
        )
        self.velocity_encoder = Encoder(
            num_genes=self.G,
            gene_dim=self.gene_dim,
            hidden=hidden_state,
            **encoder_kwargs,
        )
        
        self.manifold_codebook = SoftVectorQuantizer(
            num_codes = self.K,
            code_dim = self.gene_dim, 
            metric = "euclidean",
            normalize = False
        )
        self.velocity_codebook = SoftVectorQuantizer(
            num_codes = self.K,
            code_dim = self.gene_dim, 
            metric = "euclidean",
            normalize = True
        )
        if not self.use_vq:
            self.manifold_codebook.requires_grad_(False)
            self.velocity_codebook.requires_grad_(False)
        # 解码器输出每个基因的一个标量（s 或 v） -> output_dim=1
        self.decoder = Decoder(latent_dim = self.gene_dim, num_genes=self.G, hidden=hidden_state)

        self.loss_fn = nn.MSELoss()
        
        self.loss_history = None
        self.dataloader = dataloader
        self.layer_states = dataloader.dataset.layer_states
        self.config = config
        self.device=device
        
        self.optimizer, self.scheduler = self.create_optimizer()
        # self.optimizer = self.create_optimizer()
        
        self.cluster_edge = cluster_edge
    
    def create_optimizer(self):
        print("创建optimizer")
        r2r_params = []
        vq_params = []
        for module in [self.manifold_encoder, self.velocity_encoder, self.decoder]:
            r2r_params.extend(module.parameters())

        for module in [self.manifold_codebook, self.velocity_codebook]:
            vq_params.extend(module.parameters())

        param_groups = [
            {'params': r2r_params, 'lr': self.config.config['train']['lr']},
            {'params': vq_params, 'lr': self.config.config['train']['lr']*5},
        ]
        optimizer = torch.optim.Adam(param_groups)
        # return optimizer
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        return optimizer, scheduler

    def forward(self, inp, predict=False, x_obs=None):
        # 输入 inp: (B, 2 * G)
        z_manifold = self.manifold_encoder(inp, layer_states = self.layer_states) 
        z_velocity = self.velocity_encoder(inp, layer_states = self.layer_states) 
        
        
        # 码本输出: (B, G, K)
        if self.use_vq:
            z_s, loss_codebook_s, code_s = self.manifold_codebook(z_manifold)
            z_v, loss_codebook_v, code_v = self.velocity_codebook(z_velocity)
        else:
            zero = z_manifold.new_zeros(())
            loss_codebook_s = {"vq": zero, "commit": zero, "perplexity": zero}
            loss_codebook_v = {"vq": zero, "commit": zero, "perplexity": zero}
            code_s = torch.zeros(z_manifold.shape[:2], dtype=torch.long, device=z_manifold.device)
            code_v = torch.zeros(z_velocity.shape[:2], dtype=torch.long, device=z_velocity.device)
            z_s, z_v = z_manifold, z_velocity
        
        # 解码器输出: (B, 2 * G), (B, 2 * G)
        z_s_flatten = z_s.reshape(-1, z_s.size(-1))  # (B*G, K)
        z_v_flatten = z_v.reshape(-1, z_v.size(-1))  # (B*G, K)
        
        x_hat, v_x, losses = self.decoder(
            z_s_flatten, 
            z_v_flatten, 
            x_obs=inp if x_obs is None else x_obs,
            layer_states = self.layer_states, 
            nonneg="penalty"
        ) # x_hat: (B, 2 * G), v_x: (B, 2 * G)
        
        losses = [losses, loss_codebook_s, loss_codebook_v]
        code_use = [code_s, code_v]
        return x_hat, v_x, losses, code_use, z_manifold
    
    def compute_dynamic_adj(self, z, k=10, sigma=1.0):
        """
        z: (B, G, D), 隐空间表示
        返回动态邻接矩阵 (B, B)
        """
        z = z.max(dim=1)[0]
        dist = torch.cdist(z, z)  # (B, B)
        
        # kNN mask
        _, topk_idx = torch.topk(dist, k=k, largest=False, dim=1)  # (B, k)
        adj = torch.zeros_like(dist).scatter_(1, topk_idx, 1)
        adj = adj + adj.t()  # 对称化
        adj = torch.clamp(adj, 0, 1)  # 去重
        
        # 高斯核权重
        sim = torch.exp(-dist / (2 * sigma ** 2))
        adj = adj * sim
        return adj

    def fit(self, adjacency_matrix, device="cpu", save_path="checkpoint", directed_adjacency_matrix=None):
        self.device = device
        self.to(device)
        self.train()

        num_cells = len(self.dataloader.dataset)
        adjacency_shape = getattr(adjacency_matrix, "shape", None)
        if adjacency_shape != (num_cells, num_cells):
            raise ValueError(
                "adjacency_matrix must match the dataset cell order and size: "
                f"expected {(num_cells, num_cells)}, got {adjacency_shape}"
            )

        if directed_adjacency_matrix is None:
            directed_adjacency_matrix = adjacency_matrix
        directed_shape = getattr(directed_adjacency_matrix, "shape", None)
        if directed_shape != (num_cells, num_cells):
            raise ValueError(
                "directed_adjacency_matrix must match the dataset cell order and size: "
                f"expected {(num_cells, num_cells)}, got {directed_shape}"
            )


        num_epochs = self.config.config["train"]["num_epochs"]
        batch_counter = 0
        best_loss = float('inf')
        
        self.config.clear_history()

        for epoch in range(num_epochs):
            # 按 epoch 统计 loss
            epoch_loss_dict = {
                "loss_ode": 0.0,
                "loss_s": 0.0,
                "loss_u": 0.0,
                "loss_m_vq": 0.0,
                "loss_v_vq": 0.0,
                "loss_smooth": 0.0,
                "loss_align": 0.0,
            }

            for i, batch in enumerate(self.dataloader):
                
                # adj = adjacency_matrices[i].to(device)

                # # 每个细胞的邻居数
                # neighbor_counts = (adj > 0).sum(dim=1)

                # # 平均邻居数
                # avg_neighbors = neighbor_counts.float().mean()

                # print(
                #     f"Batch {i}: "
                #     f"avg_neighbors={avg_neighbors:.2f}, "
                #     f"min={neighbor_counts.min().item()}, "
                #     f"max={neighbor_counts.max().item()}"
                # )
                
                
                batch, is_root, batch_indices = batch
                batch = batch.to(device)
                
                u_true, s_true = batch[:, :self.G], batch[:, self.G:]  
                x_hat, v_x, losses, code_use, z = self(batch)
                # DataLoader 会 shuffle；按本批次的真实细胞索引切邻接矩阵。
                smooth_adj = slice_batch_adjacency(adjacency_matrix, batch_indices, device=device)
                align_adj = slice_batch_adjacency(directed_adjacency_matrix, batch_indices, device=device)

                # single losses
                loss_ode = losses[0]["L_ode"]
                loss_m_vq = losses[1]["vq"]
                loss_v_vq = losses[2]["vq"]

                u_pred, s_pred = x_hat[:, :self.G], x_hat[:, self.G:]
                v_u, v_s = v_x[:, :self.G], v_x[:, self.G:]

                loss_s = compute_loss_pred(s_pred, s_true)
                loss_u = compute_loss_pred(u_pred, u_true)
                loss_smooth = (compute_loss_smooth(v_u, smooth_adj) + compute_loss_smooth(v_s, smooth_adj)) / 2
                # 有向图只用于方向对齐；无向图继续用于速度平滑。
                loss_align = (compute_loss_align_pca(v_u, u_true, align_adj) + compute_loss_align_pca(v_s, s_true, align_adj)) / 2

                w = self.config.config["train"]["loss_weight"]

                # 汇总 loss
                losses_dict = {
                    "loss_ode": loss_ode,
                    "loss_s": loss_s,
                    "loss_u": loss_u,
                    "loss_m_vq": loss_m_vq,
                    "loss_v_vq": loss_v_vq,
                    "loss_smooth": loss_smooth,
                    "loss_align": loss_align,
                }

                loss = sum(w[k] * v for k, v in losses_dict.items())

                # 更新 epoch 汇总
                for k, v in losses_dict.items():
                    epoch_loss_dict[k] += v.item()

                # optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optimizer.step()

            # Epoch 平均 loss
            num_batches = len(self.dataloader)
            for k in epoch_loss_dict:
                epoch_loss_dict[k] /= num_batches
                self.config.history[k].append(epoch_loss_dict[k])

            avg_loss = sum(epoch_loss_dict[k] * w[k] for k in epoch_loss_dict)

            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.8f}")

            # Early stop
            if self.config.config["train"]["early_stop"] and \
            avg_loss + self.config.config["train"]["stop_width"] > best_loss:
                print(f"Early stopping at epoch {epoch+1}")
                break
            else:
                best_loss = min(best_loss, avg_loss)
                
            self.scheduler.step()

            # if save_path and (epoch + 1) % 25 == 0:
            #     self.save_model(f"{save_path}/epoch_{epoch+1}.pth")

        if save_path:
            self.save_model(f"{save_path}/final.pth")
    
    @torch.no_grad()
    def predict(self, batch, align=False, device='cpu'):
        self.eval()
        batch = batch.to(device)  
        x_hat, tv_x, *_  = self(batch)  
        return x_hat.cpu(), tv_x.cpu()

    @torch.no_grad()
    def get_latent(self, batch, device='cpu'):
        self.eval()
        batch = batch.to(device)
            
        z = self.manifold_encoder(batch)  # (B, G, D)
        z_flat = z.flatten(1)  # (B, G, D) -> (B, G*D)
        return z_flat.cpu()
    
    @torch.no_grad()
    def get_codebook_usage(self):
        self.eval()
        m_counts = None  # manifold codebook 使用次数
        v_counts = None  # velocity codebook 使用次数
        for batch in self.dataloader:
            batch, _, _ = batch
            batch = batch.to(next(self.parameters()).device)  # (B, G, 2)
            
            _, _, _, code_use, _ = self(batch)
            code_s, code_v = code_use
            
            # 拉平到 (B*G,)
            code_s = code_s.reshape(-1)
            code_v = code_v.reshape(-1)
            
            # 统计使用次数
            if m_counts is None:
                m_counts = torch.bincount(code_s, minlength=self.K).to(torch.float32)
                v_counts = torch.bincount(code_v, minlength=self.K).to(torch.float32)
            else:
                m_counts += torch.bincount(code_s, minlength=self.K).to(torch.float32)
                v_counts += torch.bincount(code_v, minlength=self.K).to(torch.float32)

        # 转换为频率（概率分布）
        m_usage = m_counts / m_counts.sum()
        v_usage = v_counts / v_counts.sum()

        return m_usage, v_usage
    
    def save_model(self, save_path):
        """保存模型参数到指定路径"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self.state_dict(), save_path)
        print(f"模型参数已保存到 {save_path}")
