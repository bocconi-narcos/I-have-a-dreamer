import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.weight_init import initialize_weights

class SOTARewardPredictor(nn.Module):
    """
    Simple but effective SOTA reward predictor based on current best practices.
    Uses a clean, minimal architecture with proven techniques.
    """
    def __init__(self, latent_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Input: concatenate three latent states (current, next, target)
        input_dim = latent_dim * 3
        
        # Simple but effective architecture
        self.network = nn.Sequential(
            # Input layer with normalization
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Hidden layer
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Output layer
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Proper weight initialization for better training."""
        if isinstance(module, nn.Linear):
            # Use Kaiming initialization for ReLU
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, z_t, z_tp1, z_target):
        """
        Forward pass.
        
        Args:
            z_t: Current state latent representation (B, D)
            z_tp1: Next state latent representation (B, D) 
            z_target: Target state latent representation (B, D)
            
        Returns:
            reward: Predicted reward (B, 1)
        """
        # Concatenate the three latent representations
        x = torch.cat([z_t, z_tp1, z_target], dim=1)
        
        # Pass through network
        reward = self.network(x)
        
        return reward

class ImprovedRewardPredictor(nn.Module):
    """
    Improved MLP-based reward predictor with advanced features:
    - Layer normalization for better training stability
    - Residual connections for better gradient flow
    - Skip connections for better feature preservation
    - Advanced activation functions
    - Better weight initialization
    - Attention mechanism for state importance
    """
    def __init__(self, latent_dim, hidden_dim=512, num_layers=4, dropout=0.1, use_attention=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        
        # Input: concatenate three latent states (current, next, target)
        input_dim = latent_dim * 3
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Attention mechanism for state importance
        if use_attention:
            self.state_attention = nn.MultiheadAttention(
                embed_dim=latent_dim, 
                num_heads=8, 
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(latent_dim)
        
        # Build improved MLP layers with residual connections
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for i in range(num_layers):
            # Main layer
            layer = nn.Sequential(
                nn.LayerNorm(prev_dim),
                nn.Linear(prev_dim, hidden_dim),
                nn.GELU(),  # Better than ReLU
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            self.layers.append(layer)
            
            # Residual connection
            if prev_dim != hidden_dim:
                self.layers.append(nn.Linear(prev_dim, hidden_dim))
            else:
                self.layers.append(nn.Identity())
            
            prev_dim = hidden_dim
        
        # Output layers with skip connection
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Skip connection from input to output
        self.skip_connection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights with better initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Enhanced weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, z_t, z_tp1, z_target):
        """
        Forward pass with improved architecture.
        
        Args:
            z_t: Current state latent representation (B, D)
            z_tp1: Next state latent representation (B, D) 
            z_target: Target state latent representation (B, D)
            
        Returns:
            reward: Predicted reward (B, 1)
        """
        # Apply attention if enabled
        if self.use_attention:
            # Stack states for attention
            states = torch.stack([z_t, z_tp1, z_target], dim=1)  # (B, 3, D)
            
            # Apply self-attention
            attended_states, _ = self.state_attention(states, states, states)
            attended_states = self.attention_norm(attended_states)
            
            # Flatten attended states
            z_t, z_tp1, z_target = attended_states[:, 0], attended_states[:, 1], attended_states[:, 2]
        
        # Concatenate the three latent representations
        x = torch.cat([z_t, z_tp1, z_target], dim=1)
        
        # Input normalization
        x = self.input_norm(x)
        
        # Store input for skip connection
        x_input = x
        
        # Pass through improved MLP with residual connections
        for i in range(0, len(self.layers), 2):
            layer = self.layers[i]
            residual = self.layers[i + 1]
            
            # Main path
            x_main = layer(x)
            
            # Residual path
            x_residual = residual(x)
            
            # Combine
            x = x_main + x_residual
        
        # Output with skip connection
        x = self.output_norm(x)
        x_main = self.output_layer(x)
        x_skip = self.skip_connection(x_input)
        
        # Combine main and skip outputs
        reward = x_main + 0.1 * x_skip  # Small weight for skip connection
        
        return reward

class RewardPredictor(nn.Module):
    """
    Simple MLP-based reward predictor.
    Takes three encoded states (current, next, target) and predicts a scalar reward.
    """
    def __init__(self, latent_dim, hidden_dim=256, num_layers=3, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Input: concatenate three latent states (current, next, target)
        input_dim = latent_dim * 3
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for better training."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, z_t, z_tp1, z_target):
        """
        Forward pass.
        
        Args:
            z_t: Current state latent representation (B, D)
            z_tp1: Next state latent representation (B, D) 
            z_target: Target state latent representation (B, D)
            
        Returns:
            reward: Predicted reward (B, 1)
        """
        # Concatenate the three latent representations
        x = torch.cat([z_t, z_tp1, z_target], dim=1)
        
        # Pass through MLP
        reward = self.mlp(x)
        
        return reward

class SOTARewardLoss(nn.Module):
    """
    SOTA loss function for reward prediction.
    Uses a combination of MSE and Huber loss for robustness.
    """
    def __init__(self, mse_weight=0.7, huber_weight=0.3):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.SmoothL1Loss()
        self.mse_weight = mse_weight
        self.huber_weight = huber_weight
        
    def forward(self, pred_reward, target_reward):
        """
        Compute combined loss.
        
        Args:
            pred_reward: Predicted reward (B, 1)
            target_reward: Target reward (B)
            
        Returns:
            loss: Combined MSE + Huber loss
        """
        pred = pred_reward.squeeze(-1)
        mse = self.mse_loss(pred, target_reward)
        huber = self.huber_loss(pred, target_reward)
        
        return self.mse_weight * mse + self.huber_weight * huber

class RewardPredictorLoss(nn.Module):
    """
    Simple loss function for reward prediction.
    """
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        
    def forward(self, pred_reward, target_reward):
        """
        Compute MSE loss.
        
        Args:
            pred_reward: Predicted reward (B, 1)
            target_reward: Target reward (B)
            
        Returns:
            loss: MSE loss
        """
        return self.mse_loss(pred_reward.squeeze(-1), target_reward) 