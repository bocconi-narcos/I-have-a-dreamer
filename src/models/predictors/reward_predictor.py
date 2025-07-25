import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.weight_init import initialize_weights

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