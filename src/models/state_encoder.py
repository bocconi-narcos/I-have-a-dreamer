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
        self.latent_dim = latent_dim  # Store for access

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

        # Note: CLS token removed - we output all tokens instead

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

        # final projection for all tokens
        self.to_latent = nn.Linear(self.emb_dim, latent_dim) \
            if self.emb_dim != latent_dim else nn.Identity()
        
        # final normalization for all tokens
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
                num_unique_colors: torch.LongTensor) -> tuple:
        """
        Args:
            x: (B, H, W) ints in [-1..vocab_size-2], where -1 is padding.
            shape_h: (B,) ints in [1..H]
            shape_w: (B,) ints in [1..W]
            most_common_color, least_common_color: (B,) ints in [0..vocab_size-1]
            num_unique_colors: (B,) ints in [0..vocab_size]
        Returns:
            tuple: (tokens, causal_mask)
                - tokens: (B, seq_len, latent_dim) all token representations
                - causal_mask: (B, seq_len, seq_len) boolean causal attention mask
                  where True means mask out (prevent attention)
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

        # 3) shape + stats tokens (NO CLS token)
        row_tok = self.row_shape_embed(shape_h - 1)                   # (B, emb_dim)
        col_tok = self.col_shape_embed(shape_w - 1)                   # (B, emb_dim)
        mc_tok  = self.most_common_embed(most_common_color)           # (B, emb_dim)
        lc_tok  = self.least_common_embed(least_common_color)         # (B, emb_dim)
        uq_tok  = self.unique_count_embed(num_unique_colors)          # (B, emb_dim)

        extras = torch.stack([row_tok, col_tok, mc_tok, lc_tok, uq_tok], dim=1)  # (B,5,emb_dim)
        seq = torch.cat([extras, x_flat], dim=1)                      # (B,5+H*W,emb_dim)

        # 4) dropout
        seq = self.emb_drop(seq)

        # 5) padding mask (True = mask out)
        extras_mask = torch.zeros(B, 5, dtype=torch.bool, device=x.device)  # 5 metadata tokens always kept
        padding_mask = torch.cat([extras_mask, ~grid_mask], dim=1)          # (B,5+H*W)

        # 6) create causal mask for variable-sized grids
        # Causal mask prevents attention to positions beyond valid grid boundaries
        # For each sample, we create a mask based on actual grid dimensions
        seq_len = 5 + H * W  # 5 metadata tokens + H*W grid tokens
        causal_mask = torch.zeros(B, seq_len, seq_len, dtype=torch.bool, device=x.device)
        
        grid_start_idx = 5  # Grid tokens start after 5 metadata tokens
        
        # Create grid position indices
        grid_positions = torch.arange(H * W, device=x.device)  # (H*W,)
        
        for b in range(B):
            actual_h = shape_h[b].item()
            actual_w = shape_w[b].item()
            valid_grid_size = actual_h * actual_w
            
            # Invalid positions are those beyond the valid grid size
            # Since grid is flattened row-major, positions >= valid_grid_size are invalid
            invalid_positions = grid_positions >= valid_grid_size  # (H*W,)
            
            # For grid tokens that are invalid, mask everything (including metadata)
            invalid_token_indices = grid_start_idx + grid_positions[invalid_positions]
            for idx in invalid_token_indices:
                causal_mask[b, idx, :] = True
            
            # For valid grid tokens, they can attend to:
            # - All metadata tokens (first 5) - keep unmasked
            # - Only valid grid positions - mask invalid ones
            valid_token_indices = grid_start_idx + grid_positions[~invalid_positions]
            invalid_target_indices = grid_start_idx + grid_positions[invalid_positions]
            
            # Mask out invalid grid targets for valid tokens
            for token_idx in valid_token_indices:
                causal_mask[b, token_idx, invalid_target_indices] = True
            
            # Metadata tokens (first 5) can attend to everything
            # (They're already False, which is correct)

        # 7) apply pre-norm transformer blocks
        # Note: We use padding_mask for key_padding_mask (handles padding)
        # Causal mask would be used if we implement custom attention, but MultiheadAttention
        # uses key_padding_mask, so we return causal_mask separately for potential future use
        out = seq
        for layer in self.layers:
            out = layer(out, src_key_padding_mask=padding_mask)

        # 8) apply final normalization to all tokens
        out = self.final_norm(out)                                      # (B, seq_len, emb_dim)
        
        # 9) project to latent dimension
        tokens = self.to_latent(out)                                    # (B, seq_len, latent_dim)
        
        return tokens, causal_mask
    
    def pool_tokens(self, tokens: torch.Tensor, causal_mask: torch.Tensor = None, 
                    method: str = 'mean') -> torch.Tensor:
        """
        Helper method to pool all tokens into a single vector for backward compatibility.
        
        Args:
            tokens: (B, seq_len, latent_dim) token representations
            causal_mask: (B, seq_len, seq_len) optional causal mask
            method: 'mean' or 'first' - pooling method
        
        Returns:
            (B, latent_dim) pooled representation
        """
        if method == 'mean':
            # Mean pooling over sequence dimension
            return tokens.mean(dim=1)
        elif method == 'first':
            # Use first token (first metadata token)
            return tokens[:, 0, :]
        else:
            raise ValueError(f"Unknown pooling method: {method}")


class StateEncoderWrapper(nn.Module):
    """
    Backward-compatible wrapper for StateEncoder that returns a single pooled vector
    instead of (tokens, causal_mask) tuple.
    
    This wrapper maintains backward compatibility with existing code that expects
    StateEncoder to return (B, latent_dim) instead of (tokens, causal_mask).
    
    Usage:
        # Old way (still works):
        encoder = StateEncoderWrapper(StateEncoder(...))
        latent = encoder(...)  # Returns (B, latent_dim)
        
        # New way (direct access):
        encoder = StateEncoder(...)
        tokens, causal_mask = encoder(...)  # Returns (B, seq_len, latent_dim), (B, seq_len, seq_len)
    """
    
    def __init__(self, encoder: StateEncoder, pool_method: str = 'mean'):
        """
        Args:
            encoder: StateEncoder instance to wrap
            pool_method: 'mean' or 'first' - how to pool tokens into single vector
        """
        super().__init__()
        # Register encoder as a submodule so PyTorch knows about it
        self.add_module('encoder', encoder)
        self.pool_method = pool_method
        
        # Forward all attributes to maintain compatibility
        self.latent_dim = encoder.latent_dim
        self.emb_dim = encoder.emb_dim
        self.max_rows = encoder.max_rows
        self.max_cols = encoder.max_cols
    
    def forward(self,
                x: torch.LongTensor,
                shape_h: torch.LongTensor = None,
                shape_w: torch.LongTensor = None,
                most_common_color: torch.LongTensor = None,
                least_common_color: torch.LongTensor = None,
                num_unique_colors: torch.LongTensor = None) -> torch.Tensor:
        """
        Forward pass that returns pooled latent vector for backward compatibility.
        
        Args:
            Same as StateEncoder.forward()
        Returns:
            (B, latent_dim) pooled representation (backward compatible)
        """
        # Handle optional arguments for backward compatibility
        if shape_h is None:
            # If no shape arguments provided, create defaults
            B = x.shape[0] if x.dim() >= 2 else 1
            if x.dim() == 4 and x.shape[1] == 1:
                H, W = x.shape[2], x.shape[3]
            elif x.dim() == 3:
                H, W = x.shape[1], x.shape[2]
            else:
                H, W = self.max_rows, self.max_cols
            
            shape_h = torch.full((B,), H, dtype=torch.long, device=x.device)
            shape_w = torch.full((B,), W, dtype=torch.long, device=x.device)
            most_common_color = torch.zeros(B, dtype=torch.long, device=x.device)
            least_common_color = torch.zeros(B, dtype=torch.long, device=x.device)
            num_unique_colors = torch.ones(B, dtype=torch.long, device=x.device)
        
        # Call wrapped encoder (access via _modules to avoid __getattr__ recursion)
        encoder = self._modules['encoder']
        tokens, causal_mask = encoder(
            x, shape_h, shape_w, most_common_color, least_common_color, num_unique_colors
        )
        
        # Pool tokens to single vector
        return encoder.pool_tokens(tokens, causal_mask, method=self.pool_method)
    
    def __getattr__(self, name):
        """Forward attribute access to wrapped encoder"""
        # Avoid recursion by checking if encoder exists first
        if name == 'encoder':
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        try:
            return super().__getattribute__(name)
        except AttributeError:
            # Access encoder via _modules to avoid recursion
            encoder = self._modules.get('encoder')
            if encoder is not None:
                return getattr(encoder, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")