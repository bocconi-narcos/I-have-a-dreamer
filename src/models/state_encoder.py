import torch
import torch.nn as nn
import torch.nn.functional as F

class StateEncoder(nn.Module):
    def __init__(self,
                 image_size,            # int or tuple (H, W)
                 latent_dim: int,
                 encoder_params: dict = None):
        super().__init__()
        params = encoder_params or {}
        self.depth = params.get("depth", 2)
        self.heads = params.get("heads", 4)
        self.mlp_dim = params.get("mlp_dim", 512)
        self.emb_dim = params.get("transformer_dim", 64)
        self.pool = params.get("pool", "cls")
        self.dropout = params.get("dropout", 0.3)
        self.emb_dropout = params.get("emb_dropout", 0.3)
        self.scaled_pos = params.get("scaled_position_embeddings", True)
        self.vocab_size = params.get("colors_vocab_size", 11)
        self.padding_value = -1

        # image size
        if isinstance(image_size, int):
            H = W = image_size
        else:
            H, W = image_size
        self.max_rows = H
        self.max_cols = W

        # color/token embedding (we shift x by +1 so -1→0 is the padding idx)
        self.color_embed = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)

        # positional embeddings
        if self.scaled_pos:
            # two learned vectors
            self.pos_row_embed = nn.Parameter(torch.randn(self.emb_dim))
            self.pos_col_embed = nn.Parameter(torch.randn(self.emb_dim))
        else:
            self.pos_row_embed = nn.Embedding(self.max_rows, self.emb_dim)
            self.pos_col_embed = nn.Embedding(self.max_cols, self.emb_dim)

        # shape tokens
        self.row_shape_embed = nn.Embedding(self.max_rows, self.emb_dim)
        self.col_shape_embed = nn.Embedding(self.max_cols, self.emb_dim)

        # statistic tokens
        # most/least common color indices in [0, vocab_size-1]
        self.most_common_embed = nn.Embedding(self.vocab_size, self.emb_dim)
        self.least_common_embed = nn.Embedding(self.vocab_size, self.emb_dim)
        # num_unique_colors in [0, vocab_size]
        self.unique_count_embed = nn.Embedding(self.vocab_size + 1, self.emb_dim)

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emb_dim))

        # dropout on embeddings
        self.emb_drop = nn.Dropout(self.emb_dropout)

        # Transformer encoder
        layer = nn.TransformerEncoderLayer(
            d_model=self.emb_dim,
            nhead=self.heads,
            dim_feedforward=self.mlp_dim,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=self.depth)

        # final projection to latent_dim if needed
        if self.emb_dim != latent_dim:
            self.to_latent = nn.Linear(self.emb_dim, latent_dim)
        else:
            self.to_latent = nn.Identity()


    def forward(self,
                x: torch.LongTensor,
                shape: torch.LongTensor,
                most_common_color: torch.LongTensor,
                least_common_color: torch.LongTensor,
                num_unique_colors: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W) integers in [-1..vocab_size-2], where -1 is padding.
            shape: (B, 2) with [num_rows, num_cols], values in [1..H],[1..W].
            most_common_color, least_common_color: (B,) ints in [0..vocab_size-1].
            num_unique_colors: (B,) ints in [0..vocab_size].
        Returns:
            cls_out: (B, latent_dim) the pooled [CLS] representation.
        """
        B, H, W = x.shape

        # 1) Grid mask & shift token indices
        grid_mask = (x != self.padding_value)                   # (B, H, W)
        x_tok = (x + 1).clamp(min=0)                            # shift -1→0
        # embed colors
        x_emb = self.color_embed(x_tok)                         # (B, H, W, emb_dim)

        # 2) positional embeddings
        if self.scaled_pos:
            rows = torch.arange(1, H+1, device=x.device).unsqueeze(1)  # (H,1)
            cols = torch.arange(1, W+1, device=x.device).unsqueeze(1)  # (W,1)
            pos_row = rows * self.pos_row_embed                    # (H, emb_dim)
            pos_col = cols * self.pos_col_embed                    # (W, emb_dim)
        else:
            pos_row = self.pos_row_embed(torch.arange(H, device=x.device))  # (H, emb_dim)
            pos_col = self.pos_col_embed(torch.arange(W, device=x.device))  # (W, emb_dim)
        pos = pos_row[:, None, :] + pos_col[None, :, :]            # (H, W, emb_dim)
        x_emb = x_emb + pos.unsqueeze(0)                           # (B, H, W, emb_dim)

        # flatten grid tokens
        x_flat = x_emb.view(B, H*W, self.emb_dim)                  # (B, H*W, emb_dim)
        grid_mask = grid_mask.view(B, H*W)                         # (B, H*W)

        # 3) shape + stats + CLS tokens
        row_tok = self.row_shape_embed(shape[:, 0] - 1)            # (B, emb_dim)
        col_tok = self.col_shape_embed(shape[:, 1] - 1)            # (B, emb_dim)
        mc_tok  = self.most_common_embed(most_common_color)        # (B, emb_dim)
        lc_tok  = self.least_common_embed(least_common_color)      # (B, emb_dim)
        uq_tok  = self.unique_count_embed(num_unique_colors)       # (B, emb_dim)

        # stack all tokens: [CLS, row, col, mc, lc, uq, ...grid...]
        cls = self.cls_token.expand(B, -1, -1)                     # (B, 1, emb_dim)
        extras = torch.stack([row_tok, col_tok, mc_tok, lc_tok, uq_tok], dim=1)  # (B, 5, emb_dim)
        seq = torch.cat([cls, extras, x_flat], dim=1)              # (B, 1+5+H*W, emb_dim)

        # 4) embeddings dropout
        seq = self.emb_drop(seq)

        # 5) build padding mask for transformer: True = mask out
        # extras and CLS always kept → mask False
        extras_mask = seq.new_zeros(B, 6, dtype=torch.bool)       # (B, CLS+5)
        full_mask = torch.cat([extras_mask, ~grid_mask], dim=1)   # (B, 1+5+H*W)

        # 6) transformer expects (B, seq_len, emb) with src_key_padding_mask=(B, seq_len)
        out = self.transformer(seq, src_key_padding_mask=full_mask)

        # 7) pool CLS token
        cls_out = out[:, 0, :]                                     # (B, emb_dim)
        return self.to_latent(cls_out)                            # (B, latent_dim)
