import torch
import torch.nn as nn
from einops import repeat
from src.models.base.transformer_blocks import Transformer
from src.utils.weight_init import initialize_weights

class CustomViT(nn.Module):
    """
    Custom Vision Transformer that uses concatenated embeddings instead of summed ones.
    The input dimension is divided into: 2/8 for color, 3/8 for row position, 3/8 for column position.
    """
    def __init__(self, image_size, dim, depth, heads, mlp_dim, pool='cls', dropout=0., emb_dropout=0.):
        super().__init__()
        
        image_height, image_width = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        self.image_height = image_height
        self.image_width = image_width
        
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        
        # CLS token and positional embeddings for concatenated approach
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        
        # Positional embeddings are handled differently - we don't need them here
        # since we'll create row and column embeddings separately
        
        self.dropout = nn.Dropout(emb_dropout)
        
        self.transformer = Transformer(dim, depth, heads, dim // heads, mlp_dim, dropout)
        
        self.pool = pool
        
        self.apply(initialize_weights)
    
    def forward(self, x):
        """
        Forward pass expecting already embedded and positioned tokens.
        
        Args:
            x: torch.Tensor of shape (batch, num_patches, dim)
        """
        b, n, _ = x.shape
        
        # Add CLS token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Pool the output
        if self.pool == 'mean':
            x = x[:, 1:].mean(dim=1)  # Exclude CLS token from mean
        else:  # 'cls'
            x = x[:, 0]  # Take the CLS token
        
        return x


class StateEncoder(nn.Module):
    """
    A state encoder that processes categorical image data using a custom Vision Transformer.

    This encoder is designed for images where each pixel is a category index (from -1 to 9).
    It learns separate embeddings for color, row position, and column position, then 
    concatenates them instead of summing. The latent dimension is divided as:
    - 2/8 for color embeddings
    - 3/8 for row position embeddings  
    - 3/8 for column position embeddings
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

        # Ensure latent_dim is divisible by 8
        if latent_dim % 8 != 0:
            raise ValueError(f"latent_dim must be divisible by 8, but got {latent_dim}.")

        self._image_size_tuple = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
        if isinstance(self._image_size_tuple, list):
            self._image_size_tuple = tuple(self._image_size_tuple)

        if encoder_params is None:
            encoder_params = {}

        # --- Embedding Dimensions ---
        
        # Divide latent_dim by 8: 2/8 for color, 3/8 for row, 3/8 for col
        self.color_dim = latent_dim // 4  # 2/8 = 1/4
        self.row_dim = (3 * latent_dim) // 8
        self.col_dim = (3 * latent_dim) // 8
        
        # Ensure dimensions add up correctly
        total_dim = self.color_dim + self.row_dim + self.col_dim
        if total_dim != latent_dim:
            # Adjust to make sure they sum to latent_dim exactly
            self.col_dim = latent_dim - self.color_dim - self.row_dim

        # --- Model Architecture ---

        # 1. Learned Color Embeddings (2/8 of latent_dim)
        # For pixel values in range [-1, 9], we have 11 unique categories.
        # We shift indices by +1 (from 0 to 10) for the embedding layer.
        num_pixel_categories = 11  # (-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        self.color_embedding = nn.Embedding(
            num_embeddings=num_pixel_categories,
            embedding_dim=self.color_dim
        )

        # 2. Row Position Embeddings (3/8 of latent_dim)
        h, w = self._image_size_tuple
        self.row_embedding = nn.Embedding(
            num_embeddings=h,
            embedding_dim=self.row_dim
        )

        # 3. Column Position Embeddings (3/8 of latent_dim)
        self.col_embedding = nn.Embedding(
            num_embeddings=w,
            embedding_dim=self.col_dim
        )

        # 4. Custom ViT Backbone for processing the sequence of concatenated embeddings
        vit_params = {
            'image_size': self._image_size_tuple,
            'dim': latent_dim,  # Full latent_dim after concatenation
            'depth': encoder_params.get('depth', 6),
            'heads': encoder_params.get('heads', 8),
            'mlp_dim': encoder_params.get('mlp_dim', 1024),
            'pool': encoder_params.get('pool', 'cls'),
            'dropout': encoder_params.get('dropout', 0.),
            'emb_dropout': encoder_params.get('emb_dropout', 0.)
        }
        self.encoder_model = CustomViT(**vit_params)

        num_params = sum(p.numel() for p in self.parameters())
        print(f"[StateEncoder] Number of parameters: {num_params}")
        print(f"[StateEncoder] Embedding dimensions - Color: {self.color_dim}, Row: {self.row_dim}, Col: {self.col_dim}")

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
        pixel_values = x.squeeze(1).long()  # (B, H, W)
        # Shift indices from [-1, 9] to [0, 10] to be valid for nn.Embedding
        pixel_indices = pixel_values + 1

        # Look up color embedding for each pixel: (B, H, W) -> (B, H, W, color_dim)
        color_emb = self.color_embedding(pixel_indices)

        # --- 2. Get Position Embeddings ---
        # Create row and column indices
        row_indices = torch.arange(h, device=x.device).unsqueeze(1).expand(h, w)  # (H, W)
        col_indices = torch.arange(w, device=x.device).unsqueeze(0).expand(h, w)  # (H, W)
        
        # Expand for batch dimension
        row_indices = row_indices.unsqueeze(0).expand(b, h, w)  # (B, H, W)
        col_indices = col_indices.unsqueeze(0).expand(b, h, w)  # (B, H, W)
        
        # Get embeddings
        row_emb = self.row_embedding(row_indices)  # (B, H, W, row_dim)
        col_emb = self.col_embedding(col_indices)  # (B, H, W, col_dim)

        # --- 3. Concatenate Embeddings ---
        # Concatenate along the last dimension: (B, H, W, color_dim + row_dim + col_dim)
        x_emb = torch.cat([color_emb, row_emb, col_emb], dim=-1)

        # --- 4. Reshape into a Sequence for the Transformer ---
        # (B, H, W, D) -> (B, N, D), where N = H*W is the sequence length
        x_seq = x_emb.view(b, h * w, -1)

        # --- 5. Pass through Custom ViT ---
        latent_representation = self.encoder_model(x_seq)

        return latent_representation