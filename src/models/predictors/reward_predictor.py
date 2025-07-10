import torch
import torch.nn as nn
from src.utils.weight_init import initialize_weights
from src.models.base.transformer_blocks import Transformer

class RewardPredictor(nn.Module):
    """
    Transformer-based reward predictor.
    Takes two latent vectors (z_t, z_{t+1}), performs attention among them (seq=2),
    prepends a CLS token, adds positional encoding, and outputs a scalar reward.
    Args:
        latent_dim (int): Dimension of each latent vector
        hidden_dim (int): Hidden dimension for transformer
        transformer_depth (int): Number of transformer layers
        transformer_heads (int): Number of attention heads
        transformer_dim_head (int): Dimension per attention head
        transformer_mlp_dim (int): MLP dimension inside transformer
        dropout (float): Dropout rate
        proj_dim (int, optional): If provided, project latents to this dim
    Input:
        z_t: (B, D)
        z_tp1: (B, D)
    Output:
        (B, 1) scalar reward
    """
    def __init__(self, latent_dim, hidden_dim=128, transformer_depth=2, transformer_heads=2, transformer_dim_head=64, transformer_mlp_dim=128, dropout=0.1, proj_dim=None):
        super().__init__()
        self.proj_dim = proj_dim or latent_dim
        if latent_dim != self.proj_dim:
            self.proj = nn.Linear(latent_dim, self.proj_dim)
        else:
            self.proj = nn.Identity()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.proj_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 4, self.proj_dim))  # 4 = CLS + 3 latents
        self.transformer = Transformer(
            dim=self.proj_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            dim_head=transformer_dim_head,
            mlp_dim=transformer_mlp_dim,
            dropout=dropout
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.proj_dim),
            nn.Linear(self.proj_dim, 1)
        )
        self.apply(initialize_weights)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, z_t, z_tp1, z_target):
        # z_t, z_tp1: (B, D)
        z_t = self.proj(z_t)
        z_tp1 = self.proj(z_tp1)
        z_target = self.proj(z_target)
        x = torch.stack([z_t, z_tp1, z_target], dim=1)  # (B, 3, D)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 4, D)
        x = x + self.pos_embed  # (B, 4, D)
        x = self.transformer(x)  # (B, 4, D)
        cls_out = x[:, 0]  # (B, D)
        reward = self.mlp_head(cls_out)  # (B, 1)
        return reward 