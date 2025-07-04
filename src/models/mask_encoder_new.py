#%%
import torch
import torch.nn as nn

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
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(
            x_norm, x_norm, x_norm,
            key_padding_mask=src_key_padding_mask
        )
        x = x + self.dropout1(attn_out)
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = x + self.dropout2(mlp_out)
        return x

class MaskEncoder(nn.Module):
    def __init__(self,
                 image_size,            # int or tuple (H, W)
                 vocab_size: int,       # number of mask/object categories
                 emb_dim: int = 64,
                 depth: int = 4,
                 heads: int = 8,
                 mlp_dim: int = 512,
                 dropout: float = 0.2,
                 emb_dropout: float = 0.2,
                 padding_value: int = -1):
        super().__init__()
        if isinstance(image_size, int):
            H = W = image_size
        else:
            H, W = image_size
        self.H = H
        self.W = W
        self.emb_dim = emb_dim
        self.padding_value = padding_value

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos_row_embed = nn.Embedding(H, emb_dim)
        self.pos_col_embed = nn.Embedding(W, emb_dim)
        self.emb_drop = nn.Dropout(emb_dropout)

        self.layers = nn.ModuleList([
            PreNormTransformerBlock(
                emb_dim=emb_dim,
                heads=heads,
                mlp_dim=mlp_dim,
                dropout=dropout
            )
            for _ in range(depth)
        ])

    def forward(self, x):
        """
        x: (B, H, W) or (B, 1, H, W)
        Returns: (B, emb_dim) latent vector
        """
        if x.ndim == 4:
            x = x.squeeze(1)  # (B, H, W)
        B, H, W = x.shape

        # Embedding
        x_tok = (x + 1).clamp(min=0)  # shift -1 to 0 for padding
        x_emb = self.embedding(x_tok)  # (B, H, W, emb_dim)

        # Positional Embedding
        pos_row = self.pos_row_embed(torch.arange(H, device=x.device))
        pos_col = self.pos_col_embed(torch.arange(W, device=x.device))
        pos = pos_row[:, None, :] + pos_col[None, :, :]  # (H, W, emb_dim)
        x_emb = x_emb + pos.unsqueeze(0)  # (B, H, W, emb_dim)

        # Flatten to sequence
        x_seq = x_emb.view(B, H*W, -1)  # (B, N, emb_dim)
        x_seq = self.emb_drop(x_seq)

        # Padding mask: True for positions to mask (i.e., where x == -1)
        pad_mask = (x.view(B, H*W) == self.padding_value)  # (B, N)

        # Transformer Encoder
        for layer in self.layers:
            x_seq = layer(x_seq, src_key_padding_mask=pad_mask)

        # Pooling (mean over unmasked positions)
        valid_mask = (~pad_mask).float()  # (B, N)
        sum_latent = (x_seq * valid_mask.unsqueeze(-1)).sum(dim=1)  # (B, emb_dim)
        count_latent = valid_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)  # (B, 1)
        latent = sum_latent / count_latent  # (B, emb_dim)
        return latent
    