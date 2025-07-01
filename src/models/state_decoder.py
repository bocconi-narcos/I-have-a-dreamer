import torch
import torch.nn as nn
import torch.nn.functional as F

class PreNormTransformerBlock(nn.Module):
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

        self.latent_to_seq = nn.Linear(latent_dim, self.emb_dim)
        self.position_embed = nn.Parameter(torch.randn(1, self.max_sequence_length, self.emb_dim))

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
        self.to_shape_h = nn.Linear(self.emb_dim, self.max_rows)
        self.to_shape_w = nn.Linear(self.emb_dim, self.max_cols)
        self.to_most_common = nn.Linear(self.emb_dim, self.vocab_size)
        self.to_least_common = nn.Linear(self.emb_dim, self.vocab_size)
        self.to_unique_count = nn.Linear(self.emb_dim, self.vocab_size + 1)

        num_params = sum(p.numel() for p in self.parameters())
        print(f"[StateDecoder] Number of parameters: {num_params}")

    def forward(self, z: torch.Tensor):
        B = z.shape[0]
        latent_expanded = self.latent_to_seq(z)
        seq = latent_expanded.unsqueeze(1).expand(-1, self.max_sequence_length, -1)
        seq = seq + self.position_embed

        for layer in self.layers:
            seq = layer(seq)

        metadata_tokens = seq[:, 1:6]
        grid_tokens = seq[:, 6:]

        shape_h_logits = self.to_shape_h(metadata_tokens[:, 0])
        shape_w_logits = self.to_shape_w(metadata_tokens[:, 1])
        most_common_logits = self.to_most_common(metadata_tokens[:, 2])
        least_common_logits = self.to_least_common(metadata_tokens[:, 3])
        unique_count_logits = self.to_unique_count(metadata_tokens[:, 4])

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
