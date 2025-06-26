import torch
import torch.nn as nn
from einops import repeat

# Assuming the ViT base model is located here
from src.models.base.vit import ViT
# CNN and MLP encoders are no longer used in this specialized version.
# from src.models.base.cnn import CNNEncoder
# from src.models.base.mlp import MLPEncoder

class StateEncoder(nn.Module):
    """
    A state encoder that processes categorical image data using a Vision Transformer.

    This encoder is designed for images where each pixel is a category index (from -1 to 9).
    It learns an embedding for each category ("color embedding") and combines it with a
    learned positional embedding for each pixel. The resulting sequence of tokens is
    processed by a ViT backbone.
    """
    def __init__(self,
                 image_size,  # int or tuple (h, w)
                 input_channels: int,
                 latent_dim: int,
                 encoder_params: dict = None):
        super().__init__()

        # --- Hardcoded & Validated Parameters based on new requirements ---

        # For categorical data, we expect a single channel input.
        if input_channels != 1:
            raise ValueError(f"For categorical inputs, input_channels must be 1, but got {input_channels}.")

        # Each pixel is treated as a patch.
        patch_size = 1

        self._image_size_tuple = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        if isinstance(self._image_size_tuple, list):
            self._image_size_tuple = tuple(self._image_size_tuple)

        if encoder_params is None:
            encoder_params = {}

        # --- Model Architecture ---

        # 1. Learned Color Embeddings
        # For pixel values in range [-1, 9], we have 11 unique categories.
        # We shift indices by +1 (from 0 to 10) for the embedding layer.
        num_pixel_categories = 11  # (-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        self.color_embedding = nn.Embedding(
            num_embeddings=num_pixel_categories,
            embedding_dim=latent_dim  # Embed directly into the ViT's latent space
        )

        # 2. ViT Backbone for processing the sequence of embedded pixels
        # The ViT internally creates and manages the learned positional embeddings.
        vit_params = {
            'image_size': self._image_size_tuple,
            'patch_size': patch_size,  # Hardcoded to 1
            'channels': input_channels, # This ViT param is for its internal patcher, which we bypass.
            'num_classes': 0,          # Not used for feature extraction.
            'dim': latent_dim,
            'depth': encoder_params.get('depth', 6),
            'heads': encoder_params.get('heads', 8),
            'mlp_dim': encoder_params.get('mlp_dim', 1024),
            'pool': encoder_params.get('pool', 'cls'), # Ensures (batch, dim) output
            'dropout': encoder_params.get('dropout', 0.),
            'emb_dropout': encoder_params.get('emb_dropout', 0.)
        }
        self.encoder_model = ViT(**vit_params)

        num_params = sum(p.numel() for p in self.parameters())
        print(f"[StateEncoder] Number of parameters: {num_params}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the categorical state encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, H, W) with pixel values in [-1, 9].

        Returns:
            torch.Tensor: Latent representation of shape (B, latent_dim).
        """
        # --- Input Validation ---
        if x.shape[1] != 1:
            raise ValueError(f"Expected input with 1 channel, but got {x.shape[1]}")
        b, _, h, w = x.shape
        assert h == self._image_size_tuple[0] and w == self._image_size_tuple[1], \
            f"Input image size ({h}, {w}) doesn't match model's expected size {self._image_size_tuple}."

        # --- 1. Get Color Embeddings ---
        # Squeeze channel dim and convert to long for embedding indices
        pixel_values = x.squeeze(1).long()
        # Shift indices from [-1, 9] to [0, 10] to be valid for nn.Embedding
        pixel_indices = pixel_values + 1

        # Look up color embedding for each pixel: (B, H, W) -> (B, H, W, D)
        x_emb = self.color_embedding(pixel_indices)

        # --- 2. Reshape into a Sequence for the Transformer ---
        # (B, H, W, D) -> (B, N, D), where N = H*W is the sequence length
        x_seq = x_emb.view(b, h * w, -1)

        # --- 3. Manually Execute ViT Logic (Bypassing its Image Patcher) ---

        # Prepend the [CLS] token for classification/pooling
        cls_tokens = repeat(self.encoder_model.cls_token, '1 1 d -> b 1 d', b=b)
        x_with_cls = torch.cat((cls_tokens, x_seq), dim=1)

        # Add positional embeddings (color_embedding + pos_embedding)
        # ViT's pos_embedding shape is (1, N+1, D), which broadcasts correctly.
        x_with_pos = x_with_cls + self.encoder_model.pos_embedding
        x_dropped_out = self.encoder_model.dropout(x_with_pos)

        # Pass through the main transformer blocks
        encoded_tokens = self.encoder_model.transformer(x_dropped_out)

        # Pool the output of the transformer to get a single latent vector
        if self.encoder_model.pool == 'cls':
            latent_representation = encoded_tokens[:, 0]
        elif self.encoder_model.pool == 'mean':
            latent_representation = encoded_tokens.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling type: {self.encoder_model.pool}")

        return latent_representation