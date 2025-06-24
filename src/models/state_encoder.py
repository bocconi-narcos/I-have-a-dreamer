import torch.nn as nn
from .mlp import MLPEncoder
from .cnn import CNNEncoder
from .vit import ViT

class StateEncoder(nn.Module):
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