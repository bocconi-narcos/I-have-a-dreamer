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

        # 2. Grid Statistics Embeddings
        # For unique color count (0-10 possible colors, excluding -1)
        self.unique_count_embedding = nn.Embedding(
            num_embeddings=11,  # 0 to 10 unique colors
            embedding_dim=latent_dim
        )
        
        # For most/least common colors (0-9, since we exclude -1)
        self.most_common_color_embedding = nn.Embedding(
            num_embeddings=10,  # colors 0-9
            embedding_dim=latent_dim
        )
        
        self.least_common_color_embedding = nn.Embedding(
            num_embeddings=10,  # colors 0-9
            embedding_dim=latent_dim
        )
        
        # For unpadded grid dimensions (assuming reasonable max size)
        max_dim = max(self._image_size_tuple[0], self._image_size_tuple[1])
        self.height_embedding = nn.Embedding(
            num_embeddings=max_dim + 1,  # 0 to max_dim
            embedding_dim=latent_dim
        )
        
        self.width_embedding = nn.Embedding(
            num_embeddings=max_dim + 1,  # 0 to max_dim
            embedding_dim=latent_dim
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

    def forward(self,
                x: torch.Tensor,
                shape: torch.Tensor,
                num_colors_grid: torch.Tensor,
                most_present_color: torch.Tensor,
                least_present_color: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the categorical state encoder.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, H, W) with pixel values in [-1, 9].
            shape (torch.Tensor): Unpadded grid dimensions (B, 2) or (B, 1, 2) with (height, width).
            num_colors_grid (torch.Tensor): Number of unique colors in each grid (B,).
            most_present_color (torch.Tensor): Most common color in each grid (B,).
            least_present_color (torch.Tensor): Least common color in each grid (B,).

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

        # --- 2. Create Statistic Tokens from provided data ---
        # Handle cases where stats might come in with an extra dimension from dataloader
        if shape.dim() == 3:
            shape = shape.squeeze(1)

        unpadded_heights = shape[:, 0].long()
        unpadded_widths = shape[:, 1].long()
        
        # Create embedding tokens for statistics
        unique_count_tokens = self.unique_count_embedding(num_colors_grid.long()).unsqueeze(1)  # (B, 1, D)
        most_common_tokens = self.most_common_color_embedding(most_present_color.long()).unsqueeze(1)  # (B, 1, D)
        least_common_tokens = self.least_common_color_embedding(least_present_color.long()).unsqueeze(1)  # (B, 1, D)
        height_tokens = self.height_embedding(unpadded_heights).unsqueeze(1)  # (B, 1, D)
        width_tokens = self.width_embedding(unpadded_widths).unsqueeze(1)  # (B, 1, D)
        
        # Concatenate all statistic tokens
        stat_tokens = torch.cat([
            unique_count_tokens,
            most_common_tokens, 
            least_common_tokens,
            height_tokens,
            width_tokens
        ], dim=1)  # (B, 5, D)

        # --- 3. Reshape into a Sequence for the Transformer ---
        # (B, H, W, D) -> (B, N, D), where N = H*W is the sequence length
        x_seq = x_emb.view(b, h * w, -1)
        
        # Prepend statistic tokens to the pixel sequence
        x_seq = torch.cat([stat_tokens, x_seq], dim=1)  # (B, 5+N, D)

        # --- 4. Manually Execute ViT Logic (Bypassing its Image Patcher) ---

        # Prepend the [CLS] token for classification/pooling
        cls_tokens = repeat(self.encoder_model.cls_token, '1 1 d -> b 1 d', b=b)
        x_with_cls = torch.cat((cls_tokens, x_seq), dim=1)

        # Add positional embeddings (color_embedding + pos_embedding)
        # Note: We need to adjust pos_embedding size since we added 5 statistic tokens
        # ViT's original pos_embedding shape is (1, N+1, D) for N patches + 1 CLS token
        # Now we have: 1 CLS + 5 stats + N pixels = 6+N tokens total
        original_pos_emb = self.encoder_model.pos_embedding  # (1, original_N+1, D)
        
        # We need pos embeddings for: 1 CLS + 5 stats + H*W pixels
        required_pos_tokens = 1 + 5 + h * w
        
        if original_pos_emb.size(1) != required_pos_tokens:
            # Create extended positional embeddings
            # Use the CLS pos embedding for CLS token, then create new ones for stats,
            # then use the original patch embeddings for pixels
            cls_pos = original_pos_emb[:, 0:1, :]  # (1, 1, D)
            
            # For statistics, we can interpolate or use learnable parameters
            # Here we'll use the mean of original patch embeddings as initialization
            if original_pos_emb.size(1) > 1:
                patch_pos_mean = original_pos_emb[:, 1:, :].mean(dim=1, keepdim=True)  # (1, 1, D)
                stats_pos = patch_pos_mean.repeat(1, 5, 1)  # (1, 5, D)
                pixel_pos = original_pos_emb[:, 1:, :]  # (1, original_N, D)
            else:
                # Fallback if original pos embedding only has CLS
                stats_pos = torch.zeros(1, 5, original_pos_emb.size(-1), device=original_pos_emb.device)
                pixel_pos = torch.zeros(1, h*w, original_pos_emb.size(-1), device=original_pos_emb.device)
            
            # Combine all positional embeddings
            extended_pos_emb = torch.cat([cls_pos, stats_pos, pixel_pos], dim=1)
        else:
            extended_pos_emb = original_pos_emb
        
        x_with_pos = x_with_cls + extended_pos_emb
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

    def _calculate_grid_statistics(self, x: torch.Tensor):
        """
        DEPRECATED: Grid statistics are now provided directly to the forward method.
        
        Calculate grid statistics for each image in the batch.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, H, W)
            
        Returns:
            dict: Dictionary containing statistics for each batch item
        """
        raise NotImplementedError("This method is deprecated. Pass statistics to forward().")
        b, _, h, w = x.shape
        pixel_values = x.squeeze(1)  # (B, H, W)
        
        statistics = {
            'unique_counts': [],
            'most_common_colors': [],
            'least_common_colors': [],
            'unpadded_heights': [],
            'unpadded_widths': []
        }
        
        for i in range(b):
            grid = pixel_values[i]  # (H, W)
            
            # Find unpadded dimensions by finding the rightmost and bottommost non-(-1) pixels
            non_padding_mask = (grid != -1)
            
            if non_padding_mask.any():
                # Find the last row and column that contain non-(-1) values
                rows_with_content = non_padding_mask.any(dim=1)  # (H,)
                cols_with_content = non_padding_mask.any(dim=0)  # (W,)
                
                unpadded_height = rows_with_content.nonzero(as_tuple=False).max().item() + 1
                unpadded_width = cols_with_content.nonzero(as_tuple=False).max().item() + 1
            else:
                # Edge case: entire grid is -1
                unpadded_height = 0
                unpadded_width = 0
            
            # Extract non-padding values for color statistics
            non_padding_values = grid[non_padding_mask]
            
            if len(non_padding_values) > 0:
                # Get unique colors and their counts
                unique_colors, counts = torch.unique(non_padding_values, return_counts=True)
                
                # Number of unique colors
                num_unique = len(unique_colors)
                
                # Most and least common colors
                max_count_idx = counts.argmax()
                min_count_idx = counts.argmin()
                most_common = unique_colors[max_count_idx].item()
                least_common = unique_colors[min_count_idx].item()
            else:
                # Edge case: no non-padding values
                num_unique = 0
                most_common = 0  # Default to color 0
                least_common = 0  # Default to color 0
            
            statistics['unique_counts'].append(num_unique)
            statistics['most_common_colors'].append(most_common)
            statistics['least_common_colors'].append(least_common)
            statistics['unpadded_heights'].append(unpadded_height)
            statistics['unpadded_widths'].append(unpadded_width)
        
        # Convert to tensors
        for key in statistics:
            statistics[key] = torch.tensor(statistics[key], device=x.device, dtype=torch.long)
        
        return statistics