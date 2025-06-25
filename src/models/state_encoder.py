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
            # Extract latent_dim and map it to dim for ViT
            latent_dim = kwargs.pop('latent_dim', 256)
            # Map input_channels to channels for ViT
            if 'input_channels' in kwargs:
                kwargs['channels'] = kwargs.pop('input_channels')
            # Set num_classes to 0 for feature extraction (no classification head)
            vit_kwargs = {
                'dim': latent_dim,
                'num_classes': 0,  # No classification head, return latent representation
                **kwargs
            }
            self.encoder = ViT(**vit_kwargs)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

    def forward(self, x):
        return self.encoder(x)