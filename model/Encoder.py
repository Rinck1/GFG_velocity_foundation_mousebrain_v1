import torch
import torch.nn as nn
import torch.nn.functional as F
from tool.utils import z_score

class Encoder(nn.Module):
    def __init__(
        self,
        num_genes: int,
        gene_dim: int,
        hidden=(256, 256),
        attention_heads: int = 4,
        attention_layers: int = 1,
        attention_ff_mult: int = 2,
        mlp_dropout: float = 0.2,
        attention_dropout: float = 0.1,
    ):
        super().__init__()
        if gene_dim % attention_heads != 0:
            raise ValueError(
                f"gene_dim={gene_dim} must be divisible by attention_heads={attention_heads}"
            )
        
        self.gene_dim = gene_dim
        self.num_genes = num_genes
        
        layers, d = [], 2
        for h in hidden:
            # layers.append(nn.LayerNorm(d))
            layers.append(nn.Linear(d, h))
            layers.append(nn.GELU())  
            layers.append(nn.Dropout(mlp_dropout))
            d = h
        layers += [nn.Linear(d, gene_dim)]
        self.net = nn.Sequential(*layers)
        
        # 自注意力层（作用于基因维度）
        self.attention_layer = nn.TransformerEncoderLayer(
            d_model=gene_dim,
            nhead=attention_heads,
            dim_feedforward=gene_dim * attention_ff_mult,
            dropout=attention_dropout,
            batch_first=True,
            norm_first=True,
        )
        self.attention = nn.TransformerEncoder(
            self.attention_layer,
            num_layers=attention_layers,
            enable_nested_tensor=False,
        )

    def forward(self, x, layer_states=None):
        # x: (B, 2 * G)
        
        B, twoG = x.shape
        G = twoG // 2

        # 分开 u 和 s
        u, s = x.split(G, dim=1)
        
        if layer_states is not None:
            u = z_score(u, layer_states["unspliced"]["mean"], layer_states["unspliced"]["std"])
            s = z_score(s, layer_states["spliced"]["mean"], layer_states["spliced"]["std"])

        # 堆叠成 (B, G, 2)
        x = torch.stack((u, s), dim=-1).reshape(-1, 2)

        # reshape 回 (B, G, D)
        z = self.net(x).reshape(B, G, self.gene_dim)
        
        z = self.attention(z)  # (B, G, D)，基因之间相互作用

        return z
