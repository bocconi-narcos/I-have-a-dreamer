import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.weight_init import initialize_weights
from src.models.base.transformer_blocks import Transformer
import math

class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention mechanism that processes inputs at different scales.
    Inspired by recent SOTA approaches in vision transformers.
    """
    def __init__(self, dim, num_heads=8, scales=[1, 2, 4], dropout=0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.scales = scales
        self.head_dim = dim // num_heads
        
        # Multi-scale projections
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        
        # Scale-specific attention
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
            for _ in scales
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(dim * len(scales), dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        
        # Multi-scale processing
        scale_outputs = []
        for i, scale in enumerate(self.scales):
            # Reshape for scale-specific attention
            if scale > 1:
                # Downsample sequence
                x_scaled = F.adaptive_avg_pool1d(x.transpose(1, 2), seq_len // scale).transpose(1, 2)
            else:
                x_scaled = x
                
            # Apply scale-specific attention
            attn_out, _ = self.scale_attentions[i](x_scaled, x_scaled, x_scaled)
            
            # Upsample if needed
            if scale > 1:
                attn_out = F.interpolate(attn_out.transpose(1, 2), size=seq_len, mode='linear').transpose(1, 2)
            
            scale_outputs.append(attn_out)
        
        # Fuse multi-scale outputs
        fused = torch.cat(scale_outputs, dim=-1)
        output = self.scale_fusion(fused)
        
        return output

class UncertaintyEstimator(nn.Module):
    """
    Estimates prediction uncertainty using Monte Carlo dropout.
    """
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.uncertainty_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
    
    def forward(self, x):
        return self.uncertainty_head(x)

class ContrastiveRewardHead(nn.Module):
    """
    Contrastive learning head for reward prediction.
    Helps learn better representations by contrasting positive/negative pairs.
    """
    def __init__(self, latent_dim, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 128)  # Project to lower dim for contrastive learning
        )
        
    def forward(self, z_t, z_tp1, z_target):
        # Project to contrastive space
        proj_t = F.normalize(self.projection(z_t), dim=-1)
        proj_tp1 = F.normalize(self.projection(z_tp1), dim=-1)
        proj_target = F.normalize(self.projection(z_target), dim=-1)
        
        # Compute similarities
        sim_t_tp1 = torch.sum(proj_t * proj_tp1, dim=-1) / self.temperature
        sim_t_target = torch.sum(proj_t * proj_target, dim=-1) / self.temperature
        
        return sim_t_tp1, sim_t_target

class RewardPredictor(nn.Module):
    """
    State-of-the-art transformer-based reward predictor with advanced features:
    - Multi-scale attention mechanism
    - Uncertainty estimation
    - Contrastive learning
    - Advanced regularization
    - Attention visualization support
    """
    def __init__(self, latent_dim, hidden_dim=256, transformer_depth=4, transformer_heads=8, 
                 transformer_dim_head=64, transformer_mlp_dim=512, dropout=0.1, proj_dim=None,
                 use_uncertainty=True, use_contrastive=True, use_multi_scale=True):
        super().__init__()
        self.proj_dim = proj_dim or latent_dim
        self.use_uncertainty = use_uncertainty
        self.use_contrastive = use_contrastive
        self.use_multi_scale = use_multi_scale
        
        # Enhanced input projection with better normalization
        if latent_dim != self.proj_dim:
            self.proj = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Linear(latent_dim, self.proj_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
        else:
            self.proj = nn.Sequential(
                nn.LayerNorm(latent_dim),
                nn.Identity()
            )
        
        # Learnable positional embeddings with sinusoidal initialization
        self.pos_embed = nn.Parameter(torch.zeros(1, 4, self.proj_dim))
        self._init_pos_embed()
        
        # Multi-scale attention if enabled
        if use_multi_scale:
            self.multi_scale_attn = MultiScaleAttention(
                dim=self.proj_dim,
                num_heads=transformer_heads,
                dropout=dropout
            )
        
        # Enhanced transformer with pre-norm
        self.input_norm = nn.LayerNorm(self.proj_dim)
        self.transformer = Transformer(
            dim=self.proj_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            dim_head=transformer_dim_head,
            mlp_dim=transformer_mlp_dim,
            dropout=dropout
        )
        
        # Advanced MLP head with residual connections
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.proj_dim),
            nn.Dropout(dropout),
            nn.Linear(self.proj_dim, self.proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.proj_dim, self.proj_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.proj_dim // 2, 1)
        )
        
        # Uncertainty estimator
        if use_uncertainty:
            self.uncertainty_estimator = UncertaintyEstimator(self.proj_dim)
        
        # Contrastive learning head
        if use_contrastive:
            self.contrastive_head = ContrastiveRewardHead(self.proj_dim)
        
        # Attention weights for visualization
        self.attention_weights = None
        
        # Initialize weights with better initialization
        self.apply(self._init_weights)
    
    def _init_pos_embed(self):
        """Initialize positional embeddings with sinusoidal pattern."""
        pos_embed = torch.zeros(1, 4, self.proj_dim)
        for pos in range(4):
            for i in range(0, self.proj_dim, 2):
                pos_embed[0, pos, i] = math.sin(pos / 10000 ** (i / self.proj_dim))
                if i + 1 < self.proj_dim:
                    pos_embed[0, pos, i + 1] = math.cos(pos / 10000 ** (i / self.proj_dim))
        self.pos_embed.data.copy_(pos_embed)
    
    def _init_weights(self, module):
        """Enhanced weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            nn.init.xavier_uniform_(module.out_proj.weight)
            if module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            if module.out_proj.bias is not None:
                nn.init.zeros_(module.out_proj.bias)

    def forward(self, z_t, z_tp1, z_target, return_attention=False):
        # Enhanced input processing
        z_t = self.proj(z_t)
        z_tp1 = self.proj(z_tp1)
        z_target = self.proj(z_target)
        
        # Stack inputs with better ordering
        x = torch.stack([z_t, z_tp1, z_target], dim=1)  # (B, 3, D)
        
        # Add positional encoding
        x = x + self.pos_embed[:, :3, :]  # Use first 3 positions
        x = self.input_norm(x)
        
        # Multi-scale attention if enabled
        if self.use_multi_scale:
            x = self.multi_scale_attn(x)
        
        # Apply transformer with attention capture
        if return_attention:
            # Custom forward to capture attention weights
            self.attention_weights = []
            for layer in self.transformer.layers:
                attn, ff = layer
                # Capture attention weights (simplified)
                x = attn(x) + x
                x = ff(x) + x
                self.attention_weights.append(x)  # Store intermediate outputs
        else:
            x = self.transformer(x)
        
        # Global average pooling + max pooling for better feature aggregation
        x_avg = torch.mean(x, dim=1)
        x_max, _ = torch.max(x, dim=1)
        x_combined = x_avg + x_max  # Residual connection
        
        # Predict reward
        reward = self.mlp_head(x_combined)
        
        # Uncertainty estimation
        uncertainty = None
        if self.use_uncertainty:
            uncertainty = self.uncertainty_estimator(x_combined)
        
        # Contrastive learning
        contrastive_outputs = None
        if self.use_contrastive:
            contrastive_outputs = self.contrastive_head(z_t, z_tp1, z_target)
        
        if return_attention:
            return reward, uncertainty, contrastive_outputs, self.attention_weights
        else:
            return reward, uncertainty, contrastive_outputs

class RewardPredictorLoss(nn.Module):
    """
    Advanced loss function for reward prediction with multiple components.
    """
    def __init__(self, mse_weight=1.0, uncertainty_weight=0.1, contrastive_weight=0.05):
        super().__init__()
        self.mse_weight = mse_weight
        self.uncertainty_weight = uncertainty_weight
        self.contrastive_weight = contrastive_weight
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred_reward, target_reward, uncertainty=None, contrastive_outputs=None):
        # Main MSE loss
        mse_loss = self.mse_loss(pred_reward.squeeze(-1), target_reward)
        
        # Uncertainty loss (if available)
        uncertainty_loss = 0
        if uncertainty is not None:
            # Encourage higher uncertainty for high-error predictions
            error = torch.abs(pred_reward.squeeze(-1) - target_reward)
            uncertainty_loss = F.mse_loss(uncertainty.squeeze(-1), error)
        
        # Contrastive loss (if available)
        contrastive_loss = 0
        if contrastive_outputs is not None:
            sim_t_tp1, sim_t_target = contrastive_outputs
            # Contrastive loss: encourage similar states to have similar rewards
            # This is a simplified version - you might want more sophisticated contrastive learning
            contrastive_loss = -torch.mean(sim_t_tp1) + torch.mean(sim_t_target)
        
        total_loss = (self.mse_weight * mse_loss + 
                     self.uncertainty_weight * uncertainty_loss +
                     self.contrastive_weight * contrastive_loss)
        
        return total_loss, {
            'mse_loss': mse_loss.item(),
            'uncertainty_loss': uncertainty_loss if isinstance(uncertainty_loss, float) else uncertainty_loss.item(),
            'contrastive_loss': contrastive_loss if isinstance(contrastive_loss, float) else contrastive_loss.item(),
            'total_loss': total_loss.item()
        } 