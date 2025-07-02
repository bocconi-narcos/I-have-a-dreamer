import torch
import torch.nn as nn
import torch.nn.functional as F

class PreNormTransformerBlock(nn.Module):
    """
    A pre-normalization transformer block.

    Args:
        emb_dim (int): The embedding dimension.
        heads (int): The number of attention heads.
        mlp_dim (int): The dimension of the MLP layer.
        dropout (float): The dropout rate.
    """
    def __init__(self, emb_dim, heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim * 2),
            nn.GELU(),
            nn.Linear(mlp_dim * 2, emb_dim),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        """
        Forward pass for the PreNormTransformerBlock.

        Args:
            x (torch.Tensor): The input tensor.
            src_key_padding_mask (torch.Tensor, optional): The source key padding mask. Defaults to None.

        Returns:
            torch.Tensor: The output tensor.
        """
        # Pre-norm self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=src_key_padding_mask
        )
        x = x + self.dropout1(attn_out)

        # Pre-norm feed-forward
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + self.dropout2(mlp_out)
        return x

class StateDecoder(nn.Module):
    """
    The StateDecoder model is a transformer-based decoder that reconstructs the state 
    (grid and metadata) from a latent vector.

    Args:
        image_size (int or tuple): The size of the input image (height, width).
        latent_dim (int): The dimension of the latent vector.
        decoder_params (dict, optional): A dictionary of parameters for the decoder. Defaults to {}.
    """
    def __init__(self,
                 image_size,
                 latent_dim: int,
                 decoder_params: dict = {}):
        super().__init__()
        params = decoder_params
        self.depth = params.get("depth", 4)
        self.heads = params.get("heads", 8)
        self.mlp_dim = params.get("mlp_dim", 512)
        self.emb_dim = params.get("transformer_dim", 64)
        self.dropout = params.get("dropout", 0.2)
        self.vocab_size = params.get("colors_vocab_size", 11)

        if isinstance(image_size, int):
            H = W = image_size
        else:
            H, W = image_size
        self.max_rows = H
        self.max_cols = W
        self.max_sequence_length = 1 + 5 + H * W  # CLS + metadata + grid

        # Project the latent vector to the transformer's embedding dimension
        self.latent_to_seq = nn.Linear(latent_dim, self.emb_dim)
        # Positional embeddings for the sequence
        self.position_embed = nn.Parameter(torch.randn(1, self.max_sequence_length, self.emb_dim))

        # Transformer blocks
        self.layers = nn.ModuleList([
            PreNormTransformerBlock(
                emb_dim=self.emb_dim,
                heads=self.heads,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout
            )
            for _ in range(self.depth)
        ])

        # Output heads for predicting grid and metadata
        self.to_grid = nn.Linear(self.emb_dim, self.vocab_size)
        self.to_shape_h = nn.Linear(self.emb_dim, self.max_rows)
        self.to_shape_w = nn.Linear(self.emb_dim, self.max_cols)
        self.to_most_common = nn.Linear(self.emb_dim, self.vocab_size)
        self.to_least_common = nn.Linear(self.emb_dim, self.vocab_size)
        self.to_unique_count = nn.Linear(self.emb_dim, self.vocab_size + 1)

        num_params = sum(p.numel() for p in self.parameters())
        print(f"[StateDecoder] Number of parameters: {num_params}")

    def forward(self, z: torch.Tensor):
        """
        Forward pass for the StateDecoder.

        Args:
            z (torch.Tensor): The latent vector of shape (B, latent_dim).

        Returns:
            dict: A dictionary containing the logits for the grid and metadata.
        """
        B = z.shape[0]
        # Expand the latent vector to the sequence length
        latent_expanded = self.latent_to_seq(z)
        seq = latent_expanded.unsqueeze(1).expand(-1, self.max_sequence_length, -1)
        # Add positional embeddings
        seq = seq + self.position_embed

        # Pass through transformer layers
        for layer in self.layers:
            seq = layer(seq)

        # Separate metadata and grid tokens
        metadata_tokens = seq[:, 1:6] #row, col, most_common, least_common, unique_count
        grid_tokens = seq[:, 6:] #grid

        # Project to output logits for metadata
        shape_h_logits = self.to_shape_h(metadata_tokens[:, 0]) #row
        shape_w_logits = self.to_shape_w(metadata_tokens[:, 1]) #col
        most_common_logits = self.to_most_common(metadata_tokens[:, 2]) #most_common
        least_common_logits = self.to_least_common(metadata_tokens[:, 3]) #least_common
        unique_count_logits = self.to_unique_count(metadata_tokens[:, 4]) #unique_count

        # Project to output logits for the grid
        grid_logits = self.to_grid(grid_tokens)
        grid_logits = grid_logits.view(B, self.max_rows, self.max_cols, -1)

        return {
            'grid_logits': grid_logits,
            'shape_h_logits': shape_h_logits,
            'shape_w_logits': shape_w_logits,
            'most_common_logits': most_common_logits,
            'least_common_logits': least_common_logits,
            'unique_count_logits': unique_count_logits,
        }
