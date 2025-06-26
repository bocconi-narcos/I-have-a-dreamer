import torch
import torch.nn as nn

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

