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

class StateEncoder(nn.Module):
    def __init__(self,
                 image_size,            # int or tuple (H, W)
                 input_channels: int,
                 latent_dim: int,
                 encoder_params: dict = None):
        super().__init__()
        params = encoder_params or {}
        self.depth = params.get("depth", 4)
        self.heads = params.get("heads", 8)
        self.mlp_dim = params.get("mlp_dim", 512)
        self.emb_dim = params.get("transformer_dim", 64)
        self.dropout = params.get("dropout", 0.2)
        self.emb_dropout = params.get("emb_dropout", 0.2)
        self.scaled_pos = params.get("scaled_position_embeddings", False)
        self.vocab_size = params.get("colors_vocab_size", 11)
        self.padding_value = -1

        # determine max rows/cols
        if isinstance(image_size, int):
            H = W = image_size
        else:
            H, W = image_size
        self.max_rows = H
        self.max_cols = W

        # color embedding (shift x by +1 so -1→0 is padding_idx)
        self.color_embed = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)

        # positional embeddings
        if self.scaled_pos:
            self.pos_row_embed = nn.Parameter(torch.randn(self.emb_dim))
            self.pos_col_embed = nn.Parameter(torch.randn(self.emb_dim))
        else:
            self.pos_row_embed = nn.Embedding(self.max_rows, self.emb_dim)
            self.pos_col_embed = nn.Embedding(self.max_cols, self.emb_dim)

        # shape tokens
        self.row_shape_embed = nn.Embedding(self.max_rows, self.emb_dim)
        self.col_shape_embed = nn.Embedding(self.max_cols, self.emb_dim)

        # statistic tokens
        self.most_common_embed = nn.Embedding(self.vocab_size, self.emb_dim)
        self.least_common_embed = nn.Embedding(self.vocab_size, self.emb_dim)
        self.unique_count_embed = nn.Embedding(self.vocab_size + 1, self.emb_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emb_dim))

        # dropout on embeddings
        self.emb_drop = nn.Dropout(self.emb_dropout)

        # stack of pre-norm transformer blocks
        self.layers = nn.ModuleList([
            PreNormTransformerBlock(
                emb_dim=self.emb_dim,
                heads=self.heads,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout
            )
            for _ in range(self.depth)
        ])

        # final projection
        self.to_latent = nn.Linear(self.emb_dim, latent_dim) \
            if self.emb_dim != latent_dim else nn.Identity()
        
        # final normalization for CLS token
        self.final_norm = nn.LayerNorm(self.emb_dim)
        
        # print model statistics
        num_params = sum(p.numel() for p in self.parameters())
        print(f"[StateEncoder] Number of parameters: {num_params}")

    def forward(self,
                x: torch.LongTensor,
                shape_h: torch.LongTensor,
                shape_w: torch.LongTensor,
                most_common_color: torch.LongTensor,
                least_common_color: torch.LongTensor,
                num_unique_colors: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W) ints in [-1..vocab_size-2], where -1 is padding.
            shape_h: (B,) ints in [1..H]
            shape_w: (B,) ints in [1..W]
            most_common_color, least_common_color: (B,) ints in [0..vocab_size-1]
            num_unique_colors: (B,) ints in [0..vocab_size]
        Returns:
            (B, latent_dim) pooled CLS representation.
        """

        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)  # (B, H, W)

        B, H, W = x.shape

        # 1) mask & shift tokens
        grid_mask = (x != self.padding_value)            # (B, H, W)
        x_tok = (x + 1).clamp(min=0)                     # -1→0, others shift
        x_emb = self.color_embed(x_tok)                  # (B, H, W, emb_dim)

        # 2) positional embeddings
        if self.scaled_pos:
            rows = torch.arange(1, H+1, device=x.device).unsqueeze(1)  # (H,1)
            cols = torch.arange(1, W+1, device=x.device).unsqueeze(1)  # (W,1)
            pos_row = rows * self.pos_row_embed                       # (H,emb_dim)
            pos_col = cols * self.pos_col_embed                       # (W,emb_dim)
        else:
            pos_row = self.pos_row_embed(torch.arange(H, device=x.device))
            pos_col = self.pos_col_embed(torch.arange(W, device=x.device))
        pos = pos_row[:, None, :] + pos_col[None, :, :]               # (H, W, emb_dim)
        x_emb = x_emb + pos.unsqueeze(0)                              # (B, H, W, emb_dim)

        # flatten grid
        x_flat = x_emb.view(B, H*W, self.emb_dim)                     # (B, H*W, emb_dim)
        grid_mask = grid_mask.view(B, H*W)                            # (B, H*W)

        # 3) shape + stats + CLS tokens
        row_tok = self.row_shape_embed(shape_h - 1)                   # (B, emb_dim)
        col_tok = self.col_shape_embed(shape_w - 1)                   # (B, emb_dim)
        mc_tok  = self.most_common_embed(most_common_color)           # (B, emb_dim)
        lc_tok  = self.least_common_embed(least_common_color)         # (B, emb_dim)
        uq_tok  = self.unique_count_embed(num_unique_colors)          # (B, emb_dim)

        cls = self.cls_token.expand(B, -1, -1)                        # (B,1,emb_dim)
        extras = torch.stack([row_tok, col_tok, mc_tok, lc_tok, uq_tok], dim=1)  # (B,5,emb_dim)
        seq = torch.cat([cls, extras, x_flat], dim=1)                 # (B,1+5+H*W,emb_dim)

        # 4) dropout
        seq = self.emb_drop(seq)

        # 5) padding mask (True = mask out)
        extras_mask = torch.zeros(B, 6, dtype=torch.bool, device=x.device)  # CLS+5 always kept
        full_mask = torch.cat([extras_mask, ~grid_mask], dim=1)             # (B,1+5+H*W)

        # 6) apply pre‐norm transformer blocks
        out = seq
        for layer in self.layers:
            out = layer(out, src_key_padding_mask=full_mask)

        # 7) pool CLS
        cls_out = out[:, 0, :]                                          # (B, emb_dim)
        cls_out = self.final_norm(cls_out)                              # Apply final LayerNorm
        return self.to_latent(cls_out)                                  # (B, latent_dim)
