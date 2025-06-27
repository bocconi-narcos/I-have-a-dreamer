import torch
import torch.nn as nn
from src.utils.weight_init import initialize_weights
from models.base.transformer_blocks import Transformer

class SelectionMaskPredictor(nn.Module):
    """
    Transformer-based predictor for the latent mask.
    Takes encoded_state and concat(selection_action_onehot, color_prediction) as inputs.
    Args:
        state_dim (int): Dimension of encoded_state
        selection_action_dim (int): Dimension of selection_action_onehot
        color_pred_dim (int): Dimension of color_prediction
        latent_mask_dim (int): Output dimension of the latent mask
        transformer_depth (int): Number of transformer layers
        transformer_heads (int): Number of attention heads
        transformer_dim_head (int): Dimension per attention head
        transformer_mlp_dim (int): MLP dimension inside transformer
        dropout (float): Dropout rate
    """
    def __init__(self, state_dim, selection_action_dim, color_pred_dim, latent_mask_dim,
                 transformer_depth=2, transformer_heads=2, transformer_dim_head=64, transformer_mlp_dim=128, dropout=0.1):
        super().__init__()
        self.state_dim = state_dim
        self.input2_dim = selection_action_dim + color_pred_dim
        self.latent_mask_dim = latent_mask_dim

        # Project both inputs to latent_mask_dim if needed
        self.state_proj = nn.Linear(state_dim, latent_mask_dim) if state_dim != latent_mask_dim else nn.Identity()
        self.input2_proj = nn.Linear(self.input2_dim, latent_mask_dim)

        # Positional encoding for sequence of length 2
        self.pos_embed = nn.Parameter(torch.zeros(1, 2, latent_mask_dim))

        self.transformer = Transformer(
            dim=latent_mask_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            dim_head=transformer_dim_head,
            mlp_dim=transformer_mlp_dim,
            dropout=dropout
        )
        self.output_proj = nn.Linear(latent_mask_dim, latent_mask_dim)
        self.apply(initialize_weights)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, encoded_state, selection_action_onehot, color_prediction):
        # encoded_state: (B, state_dim)
        # selection_action_onehot: (B, selection_action_dim)
        # color_prediction: (B, color_pred_dim)
        input2 = torch.cat([selection_action_onehot, color_prediction], dim=1)  # (B, input2_dim)
        state_proj = self.state_proj(encoded_state)  # (B, latent_mask_dim)
        input2_proj = self.input2_proj(input2)       # (B, latent_mask_dim)
        x = torch.stack([state_proj, input2_proj], dim=1)  # (B, 2, latent_mask_dim)
        x = x + self.pos_embed  # Add positional encoding
        x = self.transformer(x)  # (B, 2, latent_mask_dim)
        latent_mask = self.output_proj(x[:, 0])  # (B, latent_mask_dim)
        return latent_mask 