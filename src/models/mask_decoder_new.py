import torch
import torch.nn as nn

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

class MaskDecoder(nn.Module):
    """
    The MaskDecoder model is a transformer-based decoder that reconstructs the mask 
    (grid and metadata) from a latent vector.

    Args:
        image_size (int or tuple): The size of the input image (height, width).
        latent_dim (int): The dimension of the latent vector.
        decoder_params (dict, optional): A dictionary of parameters for the decoder. Defaults to {}.
        padding_value (int, optional): The value to use for padding. Defaults to -1.
    """
    def __init__(self,
                 image_size,
                 latent_dim: int,
                 decoder_params: dict = {},
                 padding_value: int = -1):
        super().__init__()
        params = decoder_params
        self.depth = params.get("depth", 4)
        self.heads = params.get("heads", 8)
        self.mlp_dim = params.get("mlp_dim", 512)
        self.emb_dim = params.get("transformer_dim", 64)
        self.dropout = params.get("dropout", 0.2)
        self.vocab_size = params.get("colors_vocab_size", 11)
        self.padding_value = padding_value

        if isinstance(image_size, int):
            H = W = image_size
        else:
            H, W = image_size
        self.max_rows = H
        self.max_cols = W
        self.max_sequence_length = H * W  # Only grid tokens

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

        self.to_grid = nn.Linear(self.emb_dim, self.vocab_size)

        num_params = sum(p.numel() for p in self.parameters())
        print(f"[MaskDecoder] Number of parameters: {num_params}")

    def forward(self, z: torch.Tensor, output_mask: torch.Tensor = None):
        """
        Forward pass for the MaskDecoder.

        Args:
            z (torch.Tensor): The latent vector of shape (B, latent_dim).
            output_mask (torch.Tensor, optional): Boolean mask of shape (B, H, W) where True indicates a padded position.

        Returns:
            grid_logits (torch.Tensor): The logits for the grid, shape (B, H, W, vocab_size).
        """
        B = z.shape[0]
        # Expand the latent vector to the sequence length
        latent_expanded = self.latent_to_seq(z)
        seq = latent_expanded.unsqueeze(1).expand(-1, self.max_sequence_length, -1)
        # Add positional embeddings
        seq = seq + self.position_embed

        # Prepare padding mask for transformer: (B, N)
        pad_mask = None
        if output_mask is not None:
            pad_mask = output_mask.view(B, self.max_sequence_length)  # True for pad positions

        # Pass through transformer layers
        for layer in self.layers:
            seq = layer(seq, src_key_padding_mask=pad_mask)

        grid_logits = self.to_grid(seq)  # (B, N, vocab_size)
        grid_logits = grid_logits.view(B, self.max_rows, self.max_cols, self.vocab_size)
        return grid_logits

# Note: When computing the loss, use the same output_mask to ignore padded positions in the loss calculation.
    
