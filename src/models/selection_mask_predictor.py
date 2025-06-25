import torch
import torch.nn as nn
from src.utils.weight_init import initialize_weights
from .base.transformer_blocks import Transformer

class SelectionMaskPredictor(nn.Module):
    """
    Predicts a latent mask representation given the latent state and one-hot selection action.
    Can be configured as an MLP or transformer block.
    Args:
        input_dim (int): Dimension of input (latent_dim + num_selection_fns)
        latent_mask_dim (int): Output dimension of the latent mask
        hidden_dim (int): Hidden dimension for MLP
        use_transformer (bool): If True, use a transformer block; else, use MLP
        transformer_depth (int): Number of transformer layers (if transformer)
        transformer_heads (int): Number of attention heads (if transformer)
        transformer_dim_head (int): Dimension per attention head (if transformer)
        transformer_mlp_dim (int): MLP dimension inside transformer (if transformer)
        dropout (float): Dropout rate
    """
    def __init__(self, input_dim, latent_mask_dim, hidden_dim=128, use_transformer=False,
                 transformer_depth=2, transformer_heads=2, transformer_dim_head=64, transformer_mlp_dim=128, dropout=0.1):
        super().__init__()
        self.use_transformer = use_transformer
        if use_transformer:
            self.input_proj = nn.Linear(input_dim, latent_mask_dim)
            self.transformer = Transformer(
                dim=latent_mask_dim,
                depth=transformer_depth,
                heads=transformer_heads,
                dim_head=transformer_dim_head,
                mlp_dim=transformer_mlp_dim,
                dropout=dropout
            )
            self.output_proj = nn.Linear(latent_mask_dim, latent_mask_dim)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_mask_dim)
            )
        self.apply(initialize_weights)

    def forward(self, x):
        if self.use_transformer:
            x = self.input_proj(x)
            x = x.unsqueeze(1)  # (B, 1, latent_mask_dim)
            x = self.transformer(x)
            x = x.squeeze(1)
            x = self.output_proj(x)
            return x  # (B, latent_mask_dim)
        else:
            return self.mlp(x)  # (B, latent_mask_dim) 