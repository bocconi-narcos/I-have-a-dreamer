import torch
import torch.nn as nn

class ColorPredictor(nn.Module):
    def __init__(self, latent_dim, num_colors=10, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_colors)
        )

    def forward(self, latent):
        return self.mlp(latent)

