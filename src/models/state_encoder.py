import torch
import torch.nn as nn
import torch.nn.functional as F


class PreNormTransformerBlock(nn.Module):
    """Same as before – kept untouched."""

    def __init__(self, emb_dim, heads, mlp_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
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
        # Pre‑norm self‑attention
        x_n = self.norm1(x)
        attn_out, _ = self.attn(x_n, x_n, x_n, key_padding_mask=src_key_padding_mask)
        x = x + self.dropout1(attn_out)

        # Pre‑norm MLP
        x_n = self.norm2(x)
        mlp_out = self.mlp(x_n)
        x = x + self.dropout2(mlp_out)
        return x


class StateEncoder(nn.Module):
    """StateEncoder with optional CNN path feeding extra tokens to the transformer.

    Args:
        image_size (int | tuple[int, int]): Input H, W.
        input_channels (int): Channel count for CNN path (set 1 if you pass index grid).
        latent_dim (int): Final latent size to project the CLS output.
        encoder_params (dict): Same keys as before **plus**:
            conv_dims (list[int]): Number of output channels for every conv layer.
            conv_stride (int): Stride for every conv layer (defaults to 2).
            conv_kernel (int): Kernel size for every conv layer (defaults to 3).
            use_conv_path (bool): Enable CNN path (default True).
    """

    def __init__(
        self,
        image_size,
        input_channels: int,
        latent_dim: int,
        encoder_params: dict | None = None,
    ) -> None:
        super().__init__()
        p = encoder_params or {}
        # ╭────────────────────────────────── Transformer params ──────────────────────────────────╮
        self.depth = p.get("depth", 4)
        self.heads = p.get("heads", 8)
        self.mlp_dim = p.get("mlp_dim", 512)
        self.emb_dim = p.get("transformer_dim", 64)
        self.dropout = p.get("dropout", 0.2)
        self.emb_dropout = p.get("emb_dropout", 0.2)
        self.scaled_pos = p.get("scaled_position_embeddings", False)
        self.vocab_size = p.get("colors_vocab_size", 11)
        self.padding_value = -1
        # ╰───────────────────────────────────────────────────────────────────────────────────────╯

        # ╭─────────────────────────────────── CNN hyper‑params ──────────────────────────────────╮
        self.use_conv_path = p.get("use_conv_path", True)
        conv_dims = p.get("conv_dims", [32, 64])
        conv_stride = p.get("conv_stride", 2)
        conv_kernel = p.get("conv_kernel", 3)
        # ╰───────────────────────────────────────────────────────────────────────────────────────╯

        # Determine H, W
        if isinstance(image_size, int):
            H = W = image_size
        else:
            H, W = image_size
        self.max_rows = H
        self.max_cols = W

        # ─────────────────────────── Token Embeddings (pixel‑wise path) ──────────────────────────
        # Color lookup (shift by +1 to reserve 0 for padding)
        self.color_embed = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)

        # Positional embeddings for every grid location
        if self.scaled_pos:
            self.pos_row_embed = nn.Parameter(torch.randn(self.emb_dim))
            self.pos_col_embed = nn.Parameter(torch.randn(self.emb_dim))
        else:
            self.pos_row_embed = nn.Embedding(self.max_rows, self.emb_dim)
            self.pos_col_embed = nn.Embedding(self.max_cols, self.emb_dim)

        # Shape / statistics tokens
        self.row_shape_embed = nn.Embedding(self.max_rows, self.emb_dim)
        self.col_shape_embed = nn.Embedding(self.max_cols, self.emb_dim)
        self.most_common_embed = nn.Embedding(self.vocab_size, self.emb_dim)
        self.least_common_embed = nn.Embedding(self.vocab_size, self.emb_dim)
        self.unique_count_embed = nn.Embedding(self.vocab_size + 1, self.emb_dim)

        # CLS token shared across paths
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.emb_dim))

        # ────────────────────────────── CNN path (optional) ─────────────────────────────────────
        if self.use_conv_path:
            layers = []
            in_ch = input_channels
            for out_ch in conv_dims:
                layers.extend(
                    [
                        nn.Conv2d(in_ch, out_ch, kernel_size=conv_kernel, stride=conv_stride, padding=conv_kernel // 2),
                        nn.GELU(),
                    ]
                )
                in_ch = out_ch
            self.conv_net = nn.Sequential(*layers)  # (B, C_L, H', W')
            # 1×1 to map channels to emb_dim so that transformer dimensions match
            self.conv_to_emb = nn.Conv2d(in_ch, self.emb_dim, kernel_size=1)
        else:
            self.conv_net = self.conv_to_emb = None

        # Dropout applied to concatenated token sequence
        self.emb_drop = nn.Dropout(self.emb_dropout)

        # Transformer stack
        self.layers = nn.ModuleList(
            [
                PreNormTransformerBlock(
                    emb_dim=self.emb_dim,
                    heads=self.heads,
                    mlp_dim=self.mlp_dim,
                    dropout=self.dropout,
                )
                for _ in range(self.depth)
            ]
        )

        # Final linear projection
        self.to_latent = (
            nn.Linear(self.emb_dim, latent_dim) if self.emb_dim != latent_dim else nn.Identity()
        )

        num_params = sum(p.numel() for p in self.parameters())
        print(f"[StateEncoder] #params: {num_params / 1e6:.2f}M")

    # ──────────────────────────────────────────── Forward ───────────────────────────────────────
    def forward(
        self,
        x: torch.LongTensor,
        shape_h: torch.LongTensor,
        shape_w: torch.LongTensor,
        most_common_color: torch.LongTensor,
        least_common_color: torch.LongTensor,
        num_unique_colors: torch.LongTensor,
    ) -> torch.Tensor:
        """Encode the input grid + global statistics into a latent vector.

        Args:
            x: (B, H, W) int grid of color IDs. Fill‑value = ‑1.
            shape_h / shape_w: (#rows, #cols) of the valid sub‑grid.
        Returns:
            (B, latent_dim) tensor.
        """

        B, H, W = x.shape
        device = x.device

        # ─── Pixel‑wise token embeddings ─────────────────────────────────────────────────────────
        grid_mask = x != self.padding_value  # (B,H,W)
        x_tok = (x + 1).clamp(min=0)  # shift padding to 0
        px_emb = self.color_embed(x_tok)  # (B,H,W,emb)

        # Positional encodings for pixel tokens
        if self.scaled_pos:
            rows = torch.arange(H, device=device).unsqueeze(1) + 1  # 1..H
            cols = torch.arange(W, device=device).unsqueeze(1) + 1  # 1..W
            pos_row = rows * self.pos_row_embed  # (H,emb)
            pos_col = cols * self.pos_col_embed  # (W,emb)
        else:
            pos_row = self.pos_row_embed(torch.arange(H, device=device))
            pos_col = self.pos_col_embed(torch.arange(W, device=device))
        pos = pos_row[:, None, :] + pos_col[None, :, :]  # (H,W,emb)
        px_emb = px_emb + pos.unsqueeze(0)
        px_tokens = px_emb.view(B, H * W, self.emb_dim)
        px_mask = (~grid_mask).view(B, H * W)  # True where pad

        # ─── CNN path (coarser tokens) ───────────────────────────────────────────────────────────
        if self.use_conv_path:
            # Prepare float image for CNN – just treat shifted token ids as grayscale
            img = x_tok.float().unsqueeze(1)  # (B,1,H,W)
            conv_feat = self.conv_net(img)  # (B,C_L,H',W')
            conv_feat = self.conv_to_emb(conv_feat)  # (B,emb,H',W')
            _, _, Hp, Wp = conv_feat.shape
            conv_tokens = conv_feat.permute(0, 2, 3, 1).reshape(B, Hp * Wp, self.emb_dim)  # (B,H'W',emb)
            # Positional encodings for conv grid (reuse row/col embeddings via interpolation)
            if self.scaled_pos:
                r = torch.linspace(0, H - 1, Hp, device=device).long()
                c = torch.linspace(0, W - 1, Wp, device=device).long()
                pos_conv = (
                    self.pos_row_embed[r][:, None, :] + self.pos_col_embed[c][None, :, :]
                )  # (H',W',emb) or scaled variant
            else:
                r = torch.linspace(0, H - 1, Hp, device=device).long()
                c = torch.linspace(0, W - 1, Wp, device=device).long()
                pos_conv = self.pos_row_embed(r)[:, None, :] + self.pos_col_embed(c)[None, :, :]
            conv_tokens = conv_tokens + pos_conv.reshape(1, Hp * Wp, self.emb_dim)
            conv_mask = torch.zeros(B, Hp * Wp, dtype=torch.bool, device=device)
        else:
            conv_tokens = torch.empty(B, 0, self.emb_dim, device=device)
            conv_mask = torch.empty(B, 0, dtype=torch.bool, device=device)

        # ─── Global statistics & CLS ──────────────────────────────────────────────────────────────
        row_tok = self.row_shape_embed(shape_h - 1)
        col_tok = self.col_shape_embed(shape_w - 1)
        mc_tok = self.most_common_embed(most_common_color)
        lc_tok = self.least_common_embed(least_common_color)
        uq_tok = self.unique_count_embed(num_unique_colors)
        extras = torch.stack([row_tok, col_tok, mc_tok, lc_tok, uq_tok], dim=1)

        cls = self.cls_token.expand(B, -1, -1)

        # Concatenate: CLS | extras | CNN | pixel
        seq = torch.cat([cls, extras, conv_tokens, px_tokens], dim=1)
        seq = self.emb_drop(seq)

        # Compose padding mask: (True = masked)
        extras_mask = torch.zeros(B, 6, dtype=torch.bool, device=device)
        full_mask = torch.cat([extras_mask, conv_mask, px_mask], dim=1)

        # ─── Transformer stack ────────────────────────────────────────────────────────────────────
        out = seq
        for layer in self.layers:
            out = layer(out, src_key_padding_mask=full_mask)

        cls_out = out[:, 0, :]
        return self.to_latent(cls_out)
