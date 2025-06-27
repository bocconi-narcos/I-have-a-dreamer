import torch
import torch.nn as nn
from src.utils.weight_init import initialize_weights
from models.base.transformer_blocks import PreNorm, FeedForward, Attention, Transformer

class NextStatePredictor(nn.Module):
    """
    Transformer-based predictor for the next latent state (x_{t+1}).
    Inputs:
        - encoded_state: Tensor of shape (batch_size, state_dim)
        - action_transform_onehot: Tensor of shape (batch_size, num_transform_actions)
        - latent_mask: Tensor of shape (batch_size, latent_mask_dim)
    Output:
        - predicted_next_latent: Tensor of shape (batch_size, latent_dim)
    """
    def __init__(self, state_dim, num_transform_actions, latent_mask_dim=0, latent_dim=None,
                 transformer_depth=2, transformer_heads=2, transformer_dim_head=64, transformer_mlp_dim=128, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.num_transform_actions = num_transform_actions
        self.latent_mask_dim = latent_mask_dim
        self.input2_dim = num_transform_actions + latent_mask_dim
        self.latent_dim = latent_dim or state_dim  # output dim

        # Project both inputs to the same dimension if needed
        self.state_proj = nn.Linear(state_dim, self.latent_dim) if state_dim != self.latent_dim else nn.Identity()
        self.input2_proj = nn.Linear(self.input2_dim, self.latent_dim)

        # Positional encoding for sequence of length 2
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, self.latent_dim))

        self.transformer = Transformer(
            dim=self.latent_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            dim_head=transformer_dim_head,
            mlp_dim=transformer_mlp_dim,
            dropout=dropout
        )
        self.output_proj = nn.Linear(self.latent_dim, self.latent_dim)
        self.apply(initialize_weights)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, encoded_state, action_transform_onehot, latent_mask=None):
        # encoded_state: (B, state_dim)
        # action_transform_onehot: (B, num_transform_actions)
        # latent_mask: (B, latent_mask_dim) or None
        if self.latent_mask_dim > 0:
            if latent_mask is None:
                latent_mask = torch.zeros(encoded_state.size(0), self.latent_mask_dim, device=encoded_state.device)
            input2 = torch.cat([action_transform_onehot, latent_mask], dim=1)
        else:
            input2 = action_transform_onehot
        # Project to common dimension
        state_proj = self.state_proj(encoded_state)  # (B, latent_dim)
        input2_proj = self.input2_proj(input2)       # (B, latent_dim)
        # Stack as sequence
        x = torch.stack([state_proj, input2_proj], dim=1)  # (B, 2, latent_dim)
        x = x + self.pos_embed  # Add positional encoding
        x = self.transformer(x)  # (B, 2, latent_dim)
        # Use the first token (state) as the output
        predicted_next_latent = self.output_proj(x[:, 0])  # (B, latent_dim)
        return predicted_next_latent
