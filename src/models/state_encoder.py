import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Union

# ============================================================================
# Components extracted from ViTARC
# ============================================================================

class FixedAbsolutePositionalEmbedding(nn.Module):
    """Sinusoidal positional embeddings from ViTARC"""
    def __init__(self, dim):
        super().__init__()
        inv_freq                                            = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t                                                   = torch.arange(16384).type_as(inv_freq)
        sinusoid_inp                                        = torch.einsum("i , j -> i j", t, inv_freq)
        emb                                                 = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        self.embed                                          = nn.Embedding.from_pretrained(emb, freeze=True)

    def forward(self, position_ids: torch.Tensor):
        return self.embed(position_ids.long())


class ViTARCEmbedding(nn.Module):
    """
    A module for mixing input embeddings and positional embeddings according to different strategies.
    
    Supported strategies:
      - 'hardcoded_normalization'
      - 'learnable_scaling'
      - 'weighted_sum'
      - 'weighted_sum_no_norm'
      - 'learnable_scaling_vec'
      - 'weighted_sum_vec'
      - 'weighted_sum_no_norm_vec'
      - 'positional_attention'
      - 'layer_norm'
      - 'default'
    """
    def __init__(self, embed_dim: int, mixer_strategy: str):
        super().__init__()
        self.embed_dim = embed_dim
        self.mixer_strategy = mixer_strategy

        # For vector-based strategies
        if self.mixer_strategy in ['learnable_scaling_vec', 'weighted_sum_vec', 'weighted_sum_no_norm_vec']:
            self.position_scale                             = nn.Parameter(torch.ones(1, embed_dim))
            self.input_weight                               = nn.Parameter(torch.ones(1, embed_dim))
            self.position_weight                            = nn.Parameter(torch.ones(1, embed_dim))

        # For scalar-based strategies
        if self.mixer_strategy in ['learnable_scaling', 'weighted_sum', 'weighted_sum_no_norm']:
            self.position_scale                             = nn.Parameter(torch.ones(1))
            self.input_weight                               = nn.Parameter(torch.ones(1))
            self.position_weight                            = nn.Parameter(torch.ones(1))

        # For positional attention
        if self.mixer_strategy == 'positional_attention':
            self.attention                                  = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

        # For layer norm
        if self.mixer_strategy == 'layer_norm':
            self.layer_norm                                 = nn.LayerNorm(embed_dim)

    def forward(self, inputs_embeds: torch.Tensor, position_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs_embeds (torch.Tensor): [batch_size, seq_len, embed_dim]
            position_embeds (torch.Tensor): [batch_size, seq_len, embed_dim]
        Returns:
            output_embeds (torch.Tensor): [batch_size, seq_len, embed_dim]
        """
        strategy = self.mixer_strategy

        if strategy == 'hardcoded_normalization':
            inputs_embeds_norm                              = F.normalize(inputs_embeds, p=2, dim=-1)
            position_embeds_norm                            = F.normalize(position_embeds, p=2, dim=-1)
            output_embeds                                   = inputs_embeds_norm + position_embeds_norm

        elif strategy in ['learnable_scaling', 'learnable_scaling_vec']:
            scaled_position_embeds                          = self.position_scale * position_embeds
            output_embeds                                   = inputs_embeds + scaled_position_embeds

        elif strategy in ['weighted_sum', 'weighted_sum_vec']:
            inputs_embeds_norm                              = F.normalize(inputs_embeds, p=2, dim=-1)
            position_embeds_norm                            = F.normalize(position_embeds, p=2, dim=-1)
            output_embeds                                   = (self.input_weight * inputs_embeds_norm) + (self.position_weight * position_embeds_norm)

        elif strategy in ['weighted_sum_no_norm', 'weighted_sum_no_norm_vec']:
            output_embeds                                   = (self.input_weight * inputs_embeds) + (self.position_weight * position_embeds)

        elif strategy == 'positional_attention':
            attn_output, _                                  = self.attention(inputs_embeds, position_embeds, position_embeds)
            output_embeds                                   = inputs_embeds + attn_output

        elif strategy == 'layer_norm':
            combined_embeds                                 = inputs_embeds + position_embeds
            output_embeds                                   = self.layer_norm(combined_embeds)

        elif strategy == 'default':
            output_embeds                                   = inputs_embeds + position_embeds

        else:
            raise ValueError(f"Unsupported mixer_strategy: {strategy}")

        return output_embeds


class EnhancedMultiheadAttention(nn.Module):
    """Enhanced MultiheadAttention with RPE support from ViTARC"""
    def __init__(self, embed_dim, num_heads, dropout=0.0, 
                 rpe_type="Two-slope-Alibi", rpe_abs=True, 
                 grid_height=32, grid_width=32):
        super().__init__()
        self.embed_dim                                      = embed_dim
        self.num_heads                                      = num_heads
        self.dropout                                        = dropout
        self.rpe_type                                       = rpe_type
        self.rpe_abs                                        = rpe_abs
        
        # Standard attention
        self.attention                                      = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # RPE setup
        if self.rpe_type in ["Four-diag-slope-Alibi", "Two-slope-Alibi"]:
            self.device                                     = next(self.parameters()).device if list(self.parameters()) else 'cpu'
            self.slopes_l                                   = torch.Tensor(self.get_slopes(num_heads, start_exponent=1)) * -1
            self.slopes_r                                   = torch.Tensor(self.get_slopes(num_heads, start_exponent=0.5)) * -1
            
            # Pre-compute 2D relative positions
            self.distance_matrix_2D                         = self.calculate_2d_relative_positions(grid_height, grid_width)
            
    def get_slopes(self, n, start_exponent=1):
        """Generate slopes for Alibi attention"""
        def get_geometric_slopes(n, start_exponent):
            start                                           = 2 ** (-start_exponent)
            ratio                                           = 2 ** -1
            return [start * (ratio ** i) for i in range(n)]

        if math.log2(n).is_integer():
            return get_geometric_slopes(n, start_exponent)
        
        else:
            closest_power_of_2                              = 2 ** math.floor(math.log2(n))
            return (get_geometric_slopes(closest_power_of_2, start_exponent) +
                    self.get_slopes(2 * closest_power_of_2, start_exponent)[0::2][:n - closest_power_of_2])
    
    def calculate_2d_relative_positions(self, grid_height, grid_width):
        """Calculate 2D relative positions for RPE"""
        if self.rpe_type == "Four-diag-slope-Alibi":
            top_right_factor                                = 2 ** 0.25
            down_right_factor                               = 2 ** 0.25
        
        else:
            top_right_factor                                = 1.0
            down_right_factor                               = 1.0
        
        # Create grid coordinates
        x_coords, y_coords                                  = torch.meshgrid(
            torch.arange(grid_height, dtype=torch.long),
            torch.arange(grid_width, dtype=torch.long),
            indexing='ij'
        )
        
        # Flatten coordinates
        x_flat                                              = x_coords.flatten()
        y_flat                                              = y_coords.flatten()
        
        # Calculate relative positions
        num_positions                                       = grid_height * grid_width
        relative_position                                   = torch.zeros((num_positions, num_positions), dtype=torch.float)
        
        for i in range(num_positions):
            for j in range(num_positions):
                x_diff                                      = x_flat[i] - x_flat[j]
                y_diff                                      = y_flat[i] - y_flat[j]
                manhattan_distance                          = float(abs(x_diff) + abs(y_diff))
                
                # Adjust distance based on direction
                if x_diff < 0 and y_diff < 0:  # Top-right
                    manhattan_distance *= top_right_factor
                
                elif x_diff > 0 and y_diff < 0:  # Down-right
                    manhattan_distance *= down_right_factor
                    
                relative_position[i, j] = manhattan_distance
                
        return relative_position
    
    def compute_rpe_bias(self, seq_len, device):
        """Compute relative positional encoding bias"""
        if self.rpe_type not in ["Four-diag-slope-Alibi", "Two-slope-Alibi"]:
            return torch.zeros((1, self.num_heads, seq_len, seq_len), device=device)
        
        # Get relevant portion of distance matrix
        relative_position                                   = self.distance_matrix_2D[:seq_len, :seq_len].to(device)
        
        if self.rpe_abs:
            relative_position                               = torch.abs(relative_position)
        
        relative_position                                   = relative_position.unsqueeze(0).expand(self.num_heads, -1, -1)
        
        self.slopes_l                                       = self.slopes_l.to(device)
        self.slopes_r                                       = self.slopes_r.to(device)
        
        # Apply slopes
        alibi_left                                          = self.slopes_l.unsqueeze(1).unsqueeze(1) * relative_position
        alibi_right                                         = self.slopes_r.unsqueeze(1).unsqueeze(1) * relative_position
        
        values                                              = torch.triu(alibi_right) + torch.tril(alibi_left)
        values                                              = values.view(1, self.num_heads, seq_len, seq_len)
        
        return values
    
    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        # Get standard attention output
        attn_output, attn_weights                           = self.attention(query, key, value, key_padding_mask=key_padding_mask)
        
        # Apply RPE bias if enabled
        if self.rpe_type in ["Four-diag-slope-Alibi", "Two-slope-Alibi"]:
            seq_len                                         = query.size(1)
            rpe_bias                                        = self.compute_rpe_bias(seq_len, query.device)
            
            # Manual attention computation with RPE
            batch_size                                      = query.size(0)
            
            # Compute attention scores
            scores                                          = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.embed_dim / self.num_heads)
            
            # Add RPE bias
            scores                                          = scores.view(batch_size, self.num_heads, seq_len, seq_len)
            scores                                         += rpe_bias
            
            # Apply masks
            if key_padding_mask is not None:
                scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
            if attn_mask is not None:
                scores += attn_mask
            
            # Apply softmax
            attn_weights                                    = F.softmax(scores, dim=-1)
            attn_weights                                    = F.dropout(attn_weights, p=self.dropout, training=self.training)
            
            # Apply attention to values
            attn_output                                     = torch.matmul(attn_weights, value.view(batch_size, self.num_heads, seq_len, -1))
            attn_output                                     = attn_output.view(batch_size, seq_len, self.embed_dim)
        
        return attn_output, attn_weights


class PreNormTransformerBlock(nn.Module):
    def __init__(self, emb_dim, heads, mlp_dim, dropout, 
                 rpe_type="Two-slope-Alibi", rpe_abs=True, 
                 grid_height=32, grid_width=32):
        super().__init__()
        self.norm1                                          = nn.LayerNorm(emb_dim)
        self.attn                                           = EnhancedMultiheadAttention(
            embed_dim=emb_dim,
            num_heads=heads,
            dropout=dropout,
            rpe_type=rpe_type,
            rpe_abs=rpe_abs,
            grid_height=grid_height,
            grid_width=grid_width
        )
        self.dropout1                                       = nn.Dropout(dropout)

        self.norm2                                          = nn.LayerNorm(emb_dim)
        self.mlp                                            = nn.Sequential(
            nn.Linear(emb_dim, mlp_dim * 2),
            nn.GELU(),
            nn.Linear(mlp_dim * 2, emb_dim),
        )
        self.dropout2                                       = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        # Pre-norm self-attention
        x_norm                                              = self.norm1(x)
        attn_out, _                                         = self.attn(x_norm, x_norm, x_norm, key_padding_mask=src_key_padding_mask)
        x                                                   = x + self.dropout1(attn_out)

        # Pre-norm feed-forward
        x_norm                                              = self.norm2(x)
        mlp_out                                             = self.mlp(x_norm)
        x                                                   = x + self.dropout2(mlp_out)
        return x


class EnhancedStateEncoder(nn.Module):
    def __init__(self,
                 image_size,            # int or tuple (H, W)
                 input_channels: int,
                 latent_dim: int,
                 encoder_params: dict = None,
                 # ViTARC-style parameters
                 ape_type: str = "SinusoidalAPE2D",
                 rpe_type: str = "Two-slope-Alibi",
                 rpe_abs: bool = True,
                 use_OPE: bool = True,
                 ape_mixer_strategy: str = "weighted_sum_no_norm"):
        super().__init__()
        params                                              = encoder_params or {}
        self.depth                                          = params.get("depth", 4)
        self.heads                                          = params.get("heads", 8)
        self.mlp_dim                                        = params.get("mlp_dim", 512)
        self.emb_dim                                        = params.get("transformer_dim", 64)
        self.dropout                                        = params.get("dropout", 0.2)
        self.emb_dropout                                    = params.get("emb_dropout", 0.2)
        self.vocab_size                                     = params.get("colors_vocab_size", 11)
        self.padding_value                                  = -1
        
        # ViTARC-style parameters
        self.ape_type                                       = ape_type
        self.rpe_type                                       = rpe_type
        self.rpe_abs                                        = rpe_abs
        self.use_OPE                                        = use_OPE
        self.ape_mixer_strategy                             = ape_mixer_strategy

        # determine max rows/cols
        if isinstance(image_size, int):
            H = W = image_size
        
        else:
            H, W = image_size
        self.max_rows = H
        self.max_cols = W

        # color embedding (shift x by +1 so -1→0 is padding_idx)
        self.color_embed                                    = nn.Embedding(self.vocab_size, self.emb_dim, padding_idx=0)

        # ViTARC-style positional embeddings
        self.setup_positional_encodings()

        # shape tokens
        self.row_shape_embed                                = nn.Embedding(self.max_rows, self.emb_dim)
        self.col_shape_embed                                = nn.Embedding(self.max_cols, self.emb_dim)

        # statistic tokens
        self.most_common_embed                              = nn.Embedding(self.vocab_size, self.emb_dim)
        self.least_common_embed                             = nn.Embedding(self.vocab_size, self.emb_dim)
        self.unique_count_embed                             = nn.Embedding(self.vocab_size + 1, self.emb_dim)

        # CLS token
        self.cls_token                                      = nn.Parameter(torch.randn(1, 1, self.emb_dim))

        # APE mixer
        self.ape_mixer                                      = ViTARCEmbedding(self.emb_dim, self.ape_mixer_strategy)

        # dropout on embeddings
        self.emb_drop                                       = nn.Dropout(self.emb_dropout)

        # stack of enhanced transformer blocks
        self.layers                                         = nn.ModuleList([
            PreNormTransformerBlock(
                emb_dim=self.emb_dim,
                heads=self.heads,
                mlp_dim=self.mlp_dim,
                dropout=self.dropout,
                rpe_type=self.rpe_type,
                rpe_abs=self.rpe_abs,
                grid_height=self.max_rows,
                grid_width=self.max_cols
            )
            for _ in range(self.depth)
        ])

        # final projection
        self.to_latent                                      = nn.Linear(self.emb_dim, latent_dim) \
            if self.emb_dim != latent_dim else nn.Identity()
        
        # print model statistics
        num_params                                          = sum(p.numel() for p in self.parameters())
        print(f"[EnhancedStateEncoder] Number of parameters: {num_params}")

    def setup_positional_encodings(self):
        """Setup positional encodings based on ViTARC configuration"""
        if self.ape_type == "LearnedAPE":
            self.wpe                                        = nn.Embedding(2048, self.emb_dim)
            self.wpe.weight.data.normal_(mean=0.0, std=0.02)
            
        elif self.ape_type == "SinusoidalAPE":
            self.wpe                                        = FixedAbsolutePositionalEmbedding(self.emb_dim)
            
        elif self.ape_type == "SinusoidalAPE2D":
            if self.use_OPE:
                # If with OPE, reserve part of embedding for object indices
                self.wpe_obj                                = FixedAbsolutePositionalEmbedding(self.emb_dim // 2)
                self.wpe_x                                  = FixedAbsolutePositionalEmbedding(self.emb_dim // 4)
                self.wpe_y                                  = FixedAbsolutePositionalEmbedding(self.emb_dim // 4)
            else:
                # Standard 2D case
                self.wpe_x                                  = FixedAbsolutePositionalEmbedding(self.emb_dim // 2)
                self.wpe_y                                  = FixedAbsolutePositionalEmbedding(self.emb_dim // 2)
            
            # 1D fallback
            self.wpe                                        = FixedAbsolutePositionalEmbedding(self.emb_dim)

    def create_positional_embeddings(self, B, H, W, device, object_ids=None):
        """Create positional embeddings based on ViTARC approach"""
        if self.ape_type == "SinusoidalAPE2D":
            # Create 2D position coordinates
            position_ids_x                                  = torch.arange(W, device=device).repeat(H)
            position_ids_y                                  = torch.arange(H, device=device).repeat_interleave(W)
            
            # Expand for batch
            position_ids_x                                  = position_ids_x.unsqueeze(0).expand(B, -1)
            position_ids_y                                  = position_ids_y.unsqueeze(0).expand(B, -1)
            
            if self.use_OPE and object_ids is not None:
                # Create object-aware embeddings
                obj_embeds                                  = self.wpe_obj(object_ids.flatten(1))  # [B, H*W, emb_dim//2]
                pos_embeds_x                                = self.wpe_x(position_ids_x)         # [B, H*W, emb_dim//4]
                pos_embeds_y                                = self.wpe_y(position_ids_y)         # [B, H*W, emb_dim//4]
                
                # Concatenate: [obj, x, y]
                position_embeds                             = torch.cat([obj_embeds, pos_embeds_x, pos_embeds_y], dim=-1)
            
            else:
                # Standard 2D case
                pos_embeds_x                                = self.wpe_x(position_ids_x)  # [B, H*W, emb_dim//2]
                pos_embeds_y                                = self.wpe_y(position_ids_y)  # [B, H*W, emb_dim//2]
                position_embeds                             = torch.cat([pos_embeds_x, pos_embeds_y], dim=-1)
                
        elif self.ape_type in ["SinusoidalAPE", "LearnedAPE"]:
            # 1D positional embeddings
            position_ids                                    = torch.arange(H * W, device=device).unsqueeze(0).expand(B, -1)
            position_embeds                                 = self.wpe(position_ids)
        
        else:
            # No positional embeddings
            position_embeds                                 = torch.zeros(B, H * W, self.emb_dim, device=device)
            
        return position_embeds

    def forward(self,
                x: torch.LongTensor,
                shape_h: torch.LongTensor,
                shape_w: torch.LongTensor,
                most_common_color: torch.LongTensor,
                least_common_color: torch.LongTensor,
                num_unique_colors: torch.LongTensor,
                object_ids: Optional[torch.LongTensor] = None) -> torch.Tensor:
        """
        Args:
            x: (B, H, W) ints in [-1..vocab_size-2], where -1 is padding.
            shape_h: (B,) ints in [1..H]
            shape_w: (B,) ints in [1..W]
            most_common_color, least_common_color: (B,) ints in [0..vocab_size-1]
            num_unique_colors: (B,) ints in [0..vocab_size]
            object_ids: (B, H, W) object indices for OPE (optional)
        Returns:
            (B, latent_dim) pooled CLS representation.
        """
        B, H, W                                             = x.shape

        # 1) mask & shift tokens
        grid_mask                                           = (x != self.padding_value)            # (B, H, W)
        x_tok                                               = (x + 1).clamp(min=0)                     # -1→0, others shift
        x_emb                                               = self.color_embed(x_tok)                  # (B, H, W, emb_dim)

        # 2) Create positional embeddings using ViTARC approach
        position_embeds                                     = self.create_positional_embeddings(
            B, H, W, x.device, object_ids=object_ids)   # (B, H*W, emb_dim)
        
        # Reshape input embeddings to match
        x_emb                                               = x_emb.view(B, H * W, self.emb_dim)       # (B, H*W, emb_dim)
        
        # 3) Mix input and positional embeddings using ViTARC mixer
        x_emb                                               = self.ape_mixer(x_emb, position_embeds)   # (B, H*W, emb_dim)

        # flatten grid mask
        grid_mask                                           = grid_mask.view(B, H*W)               # (B, H*W)

        # 4) shape + stats + CLS tokens
        row_tok                                             = self.row_shape_embed(shape_h - 1)      # (B, emb_dim)
        col_tok                                             = self.col_shape_embed(shape_w - 1)      # (B, emb_dim)
        mc_tok                                              = self.most_common_embed(most_common_color)           # (B, emb_dim)
        lc_tok                                              = self.least_common_embed(least_common_color)         # (B, emb_dim)
        uq_tok                                              = self.unique_count_embed(num_unique_colors)          # (B, emb_dim)

        cls                                                 = self.cls_token.expand(B, -1, -1)           # (B,1,emb_dim)
        extras                                              = torch.stack([row_tok, col_tok, mc_tok, lc_tok, uq_tok], dim=1)  # (B,5,emb_dim)
        seq                                                 = torch.cat([cls, extras, x_emb], dim=1)     # (B,1+5+H*W,emb_dim)

        # 5) dropout
        seq                                                 = self.emb_drop(seq)

        # 6) padding mask (True = mask out)
        extras_mask                                         = torch.zeros(B, 6, dtype=torch.bool, device=x.device)  # CLS+5 always kept
        full_mask                                           = torch.cat([extras_mask, ~grid_mask], dim=1)             # (B,1+5+H*W)

        # 7) apply enhanced transformer blocks with RPE
        out                                                 = seq
        
        for layer in self.layers:
            out                                             = layer(out, src_key_padding_mask=full_mask)

        # 8) pool CLS
        cls_out                                             = out[:, 0, :]                           # (B, emb_dim)
        
        return self.to_latent(cls_out)                   # (B, latent_dim)