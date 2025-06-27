import torch
import torch.nn as nn
from typing import Tuple
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


