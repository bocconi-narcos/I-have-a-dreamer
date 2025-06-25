import torch.nn as nn
from .mlp import MLPEncoder
from .cnn import CNNEncoder
from .vit import ViT

class MaskEncoder(nn.Module):
    """
    Encodes a selection mask (e.g., (H, W) or (C, H, W)) into a latent vector using a configurable backend (MLP, CNN, or ViT).
    Args:
        encoder_type (str): 'mlp', 'cnn', or 'vit'.
        **kwargs: Parameters for the chosen encoder.
    """
    def __init__(self, encoder_type: str, **kwargs):
        super().__init__()
        if encoder_type == 'mlp':
            self.encoder = MLPEncoder(**kwargs)
        elif encoder_type == 'cnn':
            self.encoder = CNNEncoder(**kwargs)
        elif encoder_type == 'vit':
            self.encoder = ViT(**kwargs)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def forward(self, x):
        return self.encoder(x) 