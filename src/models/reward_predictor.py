import torch
import torch.nn as nn
from src.utils.weight_init import initialize_weights
from models.base.transformer_blocks import Transformer

class RewardPredictor(nn.Module):
    """
    Predicts the reward given the encoded state and predicted next state (both latent vectors).
    Supports both MLP and Transformer architectures.
    Args:
        input_dim (int): Dimension of a single latent vector (latent_dim)
        hidden_dim (int): Hidden dimension for MLP/Transformer
        use_transformer (bool): If True, use a transformer block over the sequence [state, next_state]; else, use MLP on concatenated input
        transformer_depth (int): Number of transformer layers (if transformer)
        transformer_heads (int): Number of attention heads (if transformer)
        transformer_dim_head (int): Dimension per attention head (if transformer)
        transformer_mlp_dim (int): MLP dimension inside transformer (if transformer)
        dropout (float): Dropout rate
    Input:
        - Transformer mode: Tensor of shape (B, 2, latent_dim), where dim 1 is [encoded_state, predicted_next_state]
        - MLP mode: Tensor of shape (B, 2*latent_dim), concatenation of encoded_state and predicted_next_state
    Output:
        - Tensor of shape (B,) (predicted reward)
    """
    def __init__(self, input_dim, hidden_dim=128, use_transformer=False,
                 transformer_depth=2, transformer_heads=2, transformer_dim_head=64, transformer_mlp_dim=128, dropout=0.1):
        super().__init__()
        self.use_transformer = use_transformer
        self.input_dim = input_dim
        if use_transformer:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.transformer = Transformer(
                dim=hidden_dim,
                depth=transformer_depth,
                heads=transformer_heads,
                dim_head=transformer_dim_head,
                mlp_dim=transformer_mlp_dim,
                dropout=dropout
            )
            self.output_proj = nn.Linear(hidden_dim, 1)
        else:
            # For MLP, input is concatenated: (B, 2*latent_dim)
            self.mlp = nn.Sequential(
                nn.Linear(2 * input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        self.apply(initialize_weights)

    def forward(self, x):
        """
        Args:
            x: (B, 2, latent_dim) if transformer, (B, 2*latent_dim) if MLP
        Returns:
            (B,) tensor of predicted reward
        """
        if self.use_transformer:
            # x: (B, 2, latent_dim)
            x = self.input_proj(x)  # (B, 2, hidden_dim)
            x = self.transformer(x)  # (B, 2, hidden_dim)
            # Pool over sequence (mean or first token)
            x = x.mean(dim=1)  # (B, hidden_dim)
            x = self.output_proj(x)  # (B, 1)
            return x.squeeze(-1)  # (B,)
        else:
            # x: (B, 2*latent_dim) (concatenated)
            return self.mlp(x).squeeze(-1)  # (B,) 