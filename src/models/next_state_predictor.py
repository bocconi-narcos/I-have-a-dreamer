import torch
import torch.nn as nn
from src.utils.weight_init import initialize_weights
from src.models.transformer_blocks import PreNorm, FeedForward, Attention, Transformer

class NextStatePredictor(nn.Module):
    """
    Transformer-based predictor for the next latent state (X_{t+1}).
    Inputs:
        - latent_state: Tensor of shape (batch_size, latent_dim)
        - action_transform_onehot: Tensor of shape (batch_size, num_transform_actions) 
        - latent_mask: Tensor of shape (batch_size, latent_mask_dim) (can be zeros/placeholder for now)
    Output:
        - predicted_next_latent: Tensor of shape (batch_size, latent_dim)
    """
    def __init__(self, latent_dim, num_transform_actions, latent_mask_dim=0, 
                 transformer_depth=2, transformer_heads=2, transformer_dim_head=64, transformer_mlp_dim=128, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_transform_actions = num_transform_actions
        self.latent_mask_dim = latent_mask_dim
        self.input_dim = latent_dim + num_transform_actions + latent_mask_dim

        # Project input to transformer dimension
        self.input_proj = nn.Linear(self.input_dim, latent_dim)

        self.transformer = Transformer(
            dim=latent_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            dim_head=transformer_dim_head,
            mlp_dim=transformer_mlp_dim,
            dropout=dropout
        )

        # Output projection to latent_dim
        self.output_proj = nn.Linear(latent_dim, latent_dim)
        self.apply(initialize_weights)

    def forward(self, latent_state, action_transform_onehot, latent_mask=None):
        # latent_state: (B, latent_dim)
        # action_transform_onehot: (B, num_transform_actions)
        # latent_mask: (B, latent_mask_dim) or None
        if self.latent_mask_dim > 0:
            if latent_mask is None:
                latent_mask = torch.zeros(latent_state.size(0), self.latent_mask_dim, device=latent_state.device)
            x = torch.cat([latent_state, action_transform_onehot, latent_mask], dim=1)
        else:
            x = torch.cat([latent_state, action_transform_onehot], dim=1)
        # Project to transformer input
        x = self.input_proj(x)  # (B, latent_dim)
        # Add sequence dimension for transformer (seq_len=1)
        x = x.unsqueeze(1)  # (B, 1, latent_dim)
        x = self.transformer(x)  # (B, 1, latent_dim)
        x = x.squeeze(1)  # (B, latent_dim)
        # Output projection
        predicted_next_latent = self.output_proj(x)  # (B, latent_dim)
        return predicted_next_latent
