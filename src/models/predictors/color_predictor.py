import torch
import torch.nn as nn
from typing import Tuple, Optional
from src.models.base.transformer_blocks import PreNorm, FeedForward, Attention, Transformer

class ColorPredictor(nn.Module):
    def __init__(self, latent_dim, num_colors=10, hidden_dim=128, action_embedding_dim=12):
        super().__init__()

        self.fc1 = nn.Linear(latent_dim + action_embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_colors)
        self.relu = nn.ReLU()

        num_params = sum(p.numel() for p in self.parameters())
        print(f"[ColorPredictor] Number of parameters: {num_params}")

    def forward(self, latent, action_embedding):
        x = torch.cat([latent, action_embedding], dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class TransformerColorPredictor(nn.Module):
    """
    Transformer-based color predictor that processes state and action embeddings
    through a self-attention mechanism to predict color classes.
    
    This module constructs a sequence from state and projected action embeddings,
    processes it through transformer layers, and outputs color class logits.
    
    Args:
        state_dim (int): Dimension of the state embedding
        action_embedding_dim (int): Dimension of the action embedding
        num_colors (int): Number of color classes to predict
        transformer_depth (int): Number of transformer layers
        transformer_heads (int): Number of attention heads in transformer
        transformer_dim_head (int): Dimension of each attention head
        transformer_mlp_dim (int): Hidden dimension of transformer MLP blocks
        transformer_dropout (float): Dropout rate for transformer layers
        mlp_hidden_dim (int): Hidden dimension for the output MLP head
    """
    
    def __init__(
        self,
        state_dim: int,
        action_embedding_dim: int,
        num_colors: int = 11,
        transformer_depth: int = 4,
        transformer_heads: int = 8,
        transformer_dim_head: int = 64,
        transformer_mlp_dim: int = 256,
        transformer_dropout: float = 0.1,
        mlp_hidden_dim: int = 128
    ):
        super().__init__()
        
        # Store dimensions for reference
        self.state_dim = state_dim
        self.action_embedding_dim = action_embedding_dim
        self.num_colors = num_colors
        
        # Action projection layer: project action embedding to state dimension
        self.action_projection = nn.Linear(action_embedding_dim, state_dim)
        
        # Transformer block for processing the sequence
        self.transformer = Transformer(
            dim=state_dim,
            depth=transformer_depth,
            heads=transformer_heads,
            dim_head=transformer_dim_head,
            mlp_dim=transformer_mlp_dim,
            dropout=transformer_dropout
        )
        
        # Output MLP head: process flattened transformer output to color logits
        # Flattened dimension = 2 * state_dim (sequence length 2)
        self.mlp_head = nn.Sequential(
            nn.Linear(2 * state_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, num_colors)
        )
        
        # Print model statistics
        num_params = sum(p.numel() for p in self.parameters())
        print(f"[TransformerColorPredictor] Number of parameters: {num_params}")
        print(f"[TransformerColorPredictor] State dim: {state_dim}, Action dim: {action_embedding_dim}")
        print(f"[TransformerColorPredictor] Transformer: {transformer_depth} layers, {transformer_heads} heads")
    
    def forward(self, state_embedding: torch.Tensor, action_embedding: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the transformer color predictor.
        
        Args:
            state_embedding (torch.Tensor): State embeddings of shape (batch_size, state_dim)
            action_embedding (torch.Tensor): Action embeddings of shape (batch_size, action_embedding_dim)
            
        Returns:
            torch.Tensor: Color class logits of shape (batch_size, num_colors)
        """
        batch_size = state_embedding.shape[0]
        
        # Step 1: Project action embedding to state dimension
        projected_action = self.action_projection(action_embedding)  # (batch_size, state_dim)
        
        # Step 2: Construct sequence by stacking state and projected action
        # Result: (batch_size, 2, state_dim) where 2 is the sequence length
        sequence = torch.stack([state_embedding, projected_action], dim=1)
        
        # Step 3: Process sequence through transformer
        # Transformer operates over the sequence dimension (length 2)
        transformer_output = self.transformer(sequence)  # (batch_size, 2, state_dim)
        
        # Step 4: Flatten the transformer output for each batch
        # Concatenate the two vectors along the feature dimension
        flattened_output = transformer_output.view(batch_size, -1)  # (batch_size, 2 * state_dim)
        
        # Step 5: Pass through MLP head to predict color class logits
        color_logits = self.mlp_head(flattened_output)  # (batch_size, num_colors)
        
        return color_logits


class PreNormCrossAttentionBlock(nn.Module):
    """
    Pre-norm cross-attention block for color prediction.
    
    Uses action embedding as query and state tokens as key/value.
    Implements pre-norm architecture for stability.
    """
    def __init__(self, latent_dim: int, heads: int = 8, mlp_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        self.norm_query = nn.LayerNorm(latent_dim)
        self.norm_kv = nn.LayerNorm(latent_dim)
        
        # Cross-attention: action (query) attends to state tokens (key/value)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-forward network
        self.norm2 = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, mlp_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim * 2, latent_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        query: torch.Tensor,  # (B, 1, latent_dim) - action embedding
        key_value: torch.Tensor,  # (B, num_token, latent_dim) - state tokens
        key_padding_mask: Optional[torch.Tensor] = None  # (B, num_token) - True = mask out
    ) -> torch.Tensor:
        """
        Forward pass through cross-attention block.
        
        Args:
            query: Action embedding as query (B, 1, latent_dim)
            key_value: State tokens as key/value (B, num_token, latent_dim)
            key_padding_mask: Padding mask for state tokens (B, num_token), True = mask out
        
        Returns:
            Updated query after cross-attention (B, 1, latent_dim)
        """
        # Pre-norm cross-attention
        query_norm = self.norm_query(query)
        kv_norm = self.norm_kv(key_value)
        
        attn_out, _ = self.cross_attn(
            query_norm, kv_norm, kv_norm,
            key_padding_mask=key_padding_mask
        )
        query = query + self.dropout1(attn_out)
        
        # Pre-norm feed-forward
        query_norm = self.norm2(query)
        mlp_out = self.mlp(query_norm)
        query = query + mlp_out
        
        return query


class CrossAttentionColorPredictor(nn.Module):
    """
    Cross-attention based color predictor.
    
    Uses cross-attention layers where action embedding (query) attends to state tokens (key/value).
    Implements pre-norm architecture for stability and uses causal mask to avoid attention on padding tokens.
    
    Args:
        latent_dim: Dimension of latent embeddings (must match state token dimensions)
        num_colors: Number of color classes to predict
        action_embedding_dim: Dimension of action embedding (will be projected to latent_dim if different)
        num_layers: Number of cross-attention layers (default: 2)
        heads: Number of attention heads (default: 8)
        mlp_dim: Hidden dimension of MLP blocks (default: 256)
        dropout: Dropout rate (default: 0.1)
        mlp_hidden_dim: Hidden dimension for final MLP head (default: 128)
    """
    def __init__(
        self,
        latent_dim: int,
        num_colors: int = 11,
        action_embedding_dim: Optional[int] = None,
        num_layers: int = 2,
        heads: int = 8,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        mlp_hidden_dim: int = 128
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_colors = num_colors
        self.num_layers = num_layers
        
        # Project action embedding to latent_dim if dimensions don't match
        if action_embedding_dim is not None and action_embedding_dim != latent_dim:
            self.action_projection = nn.Linear(action_embedding_dim, latent_dim)
        else:
            self.action_projection = None
        
        # Stack of cross-attention blocks
        self.layers = nn.ModuleList([
            PreNormCrossAttentionBlock(
                latent_dim=latent_dim,
                heads=heads,
                mlp_dim=mlp_dim,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Final prediction head
        self.prediction_head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, num_colors)
        )
        
        # Print model statistics
        num_params = sum(p.numel() for p in self.parameters())
        print(f"[CrossAttentionColorPredictor] Number of parameters: {num_params}")
        print(f"[CrossAttentionColorPredictor] Latent dim: {latent_dim}, Colors: {num_colors}")
        print(f"[CrossAttentionColorPredictor] Layers: {num_layers}, Heads: {heads}, MLP dim: {mlp_dim}")
    
    def _convert_causal_mask_to_padding_mask(
        self,
        causal_mask: torch.Tensor,
        num_tokens: int
    ) -> torch.Tensor:
        """
        Convert causal mask to padding mask for MultiheadAttention.
        
        A token is considered padding if it's completely masked (all positions are True).
        This means the token cannot attend to anything, so it's effectively padding.
        
        Args:
            causal_mask: Causal mask (B, num_tokens, num_tokens) or (B, num_tokens)
            num_tokens: Number of tokens in sequence
        
        Returns:
            padding_mask: (B, num_tokens) where True = mask out (padding token)
        """
        if causal_mask is None:
            return None
        
        if causal_mask.dim() == 2:
            # Already 1D mask: (B, num_tokens)
            # If True, token is masked (padding)
            return causal_mask
        elif causal_mask.dim() == 3:
            # 2D mask: (B, num_tokens, num_tokens)
            # A token is padding if it cannot attend to anything (all True in its row)
            # Check if all positions are masked for each token
            padding_mask = causal_mask.all(dim=-1)  # (B, num_tokens)
            return padding_mask
        else:
            raise ValueError(f"Unexpected causal_mask dimension: {causal_mask.dim()}")
    
    def forward(
        self,
        action_embedding: torch.Tensor,  # (B, action_embedding_dim) or (B, latent_dim)
        state_tokens: torch.Tensor,  # (B, num_token, latent_dim)
        causal_mask: Optional[torch.Tensor] = None  # (B, num_token, num_token) or (B, num_token)
    ) -> torch.Tensor:
        """
        Forward pass through cross-attention color predictor.
        
        Args:
            action_embedding: Action embedding (B, action_embedding_dim) or (B, latent_dim)
            state_tokens: State tokens from encoder (B, num_token, latent_dim)
            causal_mask: Causal mask indicating valid tokens (B, num_token, num_token) or (B, num_token)
                        If None, all tokens are considered valid
        
        Returns:
            color_logits: Color class logits (B, num_colors)
        """
        B = action_embedding.shape[0]
        
        # Project action embedding to latent_dim if necessary
        if self.action_projection is not None:
            action_embedding = self.action_projection(action_embedding)  # (B, latent_dim)
        
        # Expand action to sequence format: (B, latent_dim) -> (B, 1, latent_dim)
        query = action_embedding.unsqueeze(1)  # (B, 1, latent_dim)
        
        # Convert causal mask to padding mask for MultiheadAttention
        # padding_mask[i] = True means token i is padding and should be masked out
        padding_mask = self._convert_causal_mask_to_padding_mask(
            causal_mask, state_tokens.shape[1]
        )
        
        # Apply cross-attention layers
        # Each layer: action (query) attends to state tokens (key/value)
        for layer in self.layers:
            query = layer(query, state_tokens, key_padding_mask=padding_mask)
        
        # Extract action representation: (B, 1, latent_dim) -> (B, latent_dim)
        action_repr = query.squeeze(1)  # (B, latent_dim)
        
        # Final prediction head
        color_logits = self.prediction_head(action_repr)  # (B, num_colors)
        
        return color_logits


