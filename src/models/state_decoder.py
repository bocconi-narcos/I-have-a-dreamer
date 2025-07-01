import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import h5py

# class StateDecoder(nn.Module):
#     def __init__(self, latent_dim, n_attention_head, num_layers, **kwargs):
#         super().__init__()
#         decoder_layer = TransformerDecoderLayer(latent_dim, n_attention_head, **kwargs)
#         self.decoder = TransformerDecoder(decoder_layer, num_layers)
        

#     def forward(self, tgt, tgt_mask=None):
#         return self.decoder(tgt, tgt_mask=tgt_mask)


#         project latent state into transformer dim
#         set this projection as memory
#         Encoding:
#         x_t from R^d to z_t from R^d_T
#         for each pixel, take color, row, column vector from R^d, sum all of them into a vector still in R^d, stack all 9*9 = 81 tensors horizontally and with self attention create the latent dimension R^d_T
#         Decoding:
#         sum embedded_row and embedded_column, create an embedded vector in R^d_T, stack all 81 vertically, then based on the latent vector and each positional embedding (embedded row and column) predict color of specific pixel referring to that positional embedding vector.
        
#         Adjustments
#         Done: latent_dim can be different from transformer dim
#         Done: adjustable depth of decoder_layer
#         Done: adjustable MLP dimension in transformer layer
#         causal mask and/or autoregressive progression to predict (understanding the differences between the two)


# def set_device(file='file.py'):
#     if torch.cuda.is_available():
#         device = torch.device("cuda")
#     elif torch.backends.mps.is_available():
#         device = torch.device("mps")
#     else:
#         device = torch.device("cpu")

#     print(f"Using device: {device} for {file}")
#     return device

# DEVICE = set_device('state_decoder.py')


class StateDecoder(nn.Module):
    '''
    State Decoder
    '''
    def __init__(self, emb_state_dim: int, emb_dim: int, num_layers: int, max_rows: int, max_cols: int, vocab_size: int, mlp_dim: int):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_rows = max_rows
        self.max_cols = max_cols
        self.vocab_size = vocab_size
        self.N = max_rows * max_cols

        # Query embeddings for each pixel position
        self.query_embed = nn.Parameter(torch.randn(1, self.N, emb_dim))

        # Project latent to memory dimension if needed
        self.latent_proj = nn.Linear(emb_state_dim, emb_dim)

        decoder_layer = TransformerDecoderLayer(d_model=emb_dim, nhead=4, dim_feedforward=mlp_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers)

        self.layer_norm = nn.LayerNorm(emb_dim)
        self.shape_row_proj = nn.Linear(emb_dim, max_rows)
        self.shape_col_proj = nn.Linear(emb_dim, max_cols)
        self.grid_proj = nn.Linear(emb_dim, vocab_size)

    def forward(self, latent, H, W, dropout_eval: bool):
        # latent: (batch, latent_dim)
        B = latent.size(0)
        N = H * W

        # Project latent to emb_dim if needed
        memory = self.latent_proj(latent).unsqueeze(1)  # (B, 1, emb_dim)

        # Repeat query embeddings for batch
        queries = self.query_embed.repeat(B, 1, 1)  # (B, N, emb_dim)

        # Prepare 2D causal mask
        causal_mask = self._make_2d_causal_mask(device=latent.device)

        # Transformer expects (N, B, emb_dim)
        queries = queries.transpose(0, 1)  # (N, B, emb_dim)
        memory = memory.transpose(0, 1)    # (1, B, emb_dim)
        output = self.transformer_decoder(queries, memory, tgt_mask=causal_mask)

        output = output.transpose(0, 1)  # (B, N, emb_dim)
        output = self.layer_norm(output)
        shape_row_logits = self.shape_row_proj(output)
        shape_col_logits = self.shape_col_proj(output)
        grid_logits = self.grid_proj(output)  # (B, N, vocab_size)
        return shape_row_logits, shape_col_logits, grid_logits

    def _make_2d_causal_mask(self, device=None):
        H, W = self.max_rows, self.max_cols
        N = H * W
        row_indices = torch.arange(N, device=device) // W
        col_indices = torch.arange(N, device=device) % W
        row_i = row_indices.unsqueeze(1)
        row_j = row_indices.unsqueeze(0)
        col_i = col_indices.unsqueeze(1)
        col_j = col_indices.unsqueeze(0)
        mask = (row_j > row_i) | ((row_j == row_i) & (col_j > col_i))
        return mask





# #%%
# import torch
# import torch.nn as nn
# from einops import repeat

# # Assuming the ViT base model is located here
# from src.models.base.vit import ViT
# # CNN and MLP encoders are no longer used in this specialized version.
# # from src.models.base.cnn import CNNEncoder
# # from src.models.base.mlp import MLPEncoder

# class StateEncoder(nn.Module):
#     """
#     A state encoder that processes categorical image data using a Vision Transformer.

#     This encoder is designed for images where each pixel is a category index (from -1 to 9).
#     It learns an embedding for each category ("color embedding") and combines it with a
#     learned positional embedding for each pixel. The resulting sequence of tokens is
#     processed by a ViT backbone.
#     """
#     def __init__(self,
#                  image_size,  # int or tuple (h, w)
#                  input_channels: int,
#                  latent_dim: int,
#                  encoder_params: dict = None):
#         super().__init__()

#         # --- Hardcoded & Validated Parameters based on new requirements ---

#         # For categorical data, we expect a single channel input.
#         if input_channels != 1:
#             raise ValueError(f"For categorical inputs, input_channels must be 1, but got {input_channels}.")

#         # Each pixel is treated as a patch.
#         patch_size = 1

#         self._image_size_tuple = image_size if isinstance(image_size, (tuple, list)) else (image_size, image_size)
#         if isinstance(self._image_size_tuple, list):
#             self._image_size_tuple = tuple(self._image_size_tuple)

#         if encoder_params is None:
#             encoder_params = {}

#         # --- Model Architecture ---

#         # 1. Learned Color Embeddings
#         # For pixel values in range [-1, 9], we have 11 unique categories.
#         # We shift indices by +1 (from 0 to 10) for the embedding layer.
#         num_pixel_categories = 11  # (-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
#         self.color_embedding = nn.Embedding(
#             num_embeddings=num_pixel_categories,
#             embedding_dim=latent_dim  # Embed directly into the ViT's latent space
#         )

#         # 2. ViT Backbone for processing the sequence of embedded pixels
#         # The ViT internally creates and manages the learned positional embeddings.
#         vit_params = {
#             'image_size': self._image_size_tuple,
#             'patch_size': patch_size,  # Hardcoded to 1
#             'channels': input_channels, # This ViT param is for its internal patcher, which we bypass.
#             'num_classes': 0,          # Not used for feature extraction.
#             'dim': latent_dim,
#             'depth': encoder_params.get('depth', 6),
#             'heads': encoder_params.get('heads', 8),
#             'mlp_dim': encoder_params.get('mlp_dim', 1024),
#             'pool': encoder_params.get('pool', 'cls'), # Ensures (batch, dim) output
#             'dropout': encoder_params.get('dropout', 0.),
#             'emb_dropout': encoder_params.get('emb_dropout', 0.)
#         }
#         self.encoder_model = ViT(**vit_params)

#         num_params = sum(p.numel() for p in self.parameters())
#         print(f"[StateEncoder] Number of parameters: {num_params}")

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass for the categorical state encoder.

#         Args:
#             x (torch.Tensor): Input tensor of shape (B, 1, H, W) with pixel values in [-1, 9].

#         Returns:
#             torch.Tensor: Latent representation of shape (B, latent_dim).
#         """
#         # --- Input Validation ---
#         if x.shape[1] != 1:
#             raise ValueError(f"Expected input with 1 channel, but got {x.shape[1]}")
#         b, _, h, w = x.shape
#         assert h == self._image_size_tuple[0] and w == self._image_size_tuple[1], \
#             f"Input image size ({h}, {w}) doesn't match model's expected size {self._image_size_tuple}."

#         # --- 1. Get Color Embeddings ---
#         # Squeeze channel dim and convert to long for embedding indices
#         pixel_values = x.squeeze(1).long()
#         # Shift indices from [-1, 9] to [0, 10] to be valid for nn.Embedding
#         pixel_indices = pixel_values + 1

#         # Look up color embedding for each pixel: (B, H, W) -> (B, H, W, D)
#         x_emb = self.color_embedding(pixel_indices)

#         # --- 2. Reshape into a Sequence for the Transformer ---
#         # (B, H, W, D) -> (B, N, D), where N = H*W is the sequence length
#         x_seq = x_emb.view(b, h * w, -1)

#         # --- 3. Manually Execute ViT Logic (Bypassing its Image Patcher) ---

#         # Prepend the [CLS] token for classification/pooling
#         cls_tokens = repeat(self.encoder_model.cls_token, '1 1 d -> b 1 d', b=b)
#         x_with_cls = torch.cat((cls_tokens, x_seq), dim=1)

#         # Add positional embeddings (color_embedding + pos_embedding)
#         # ViT's pos_embedding shape is (1, N+1, D), which broadcasts correctly.
#         x_with_pos = x_with_cls + self.encoder_model.pos_embedding
#         x_dropped_out = self.encoder_model.dropout(x_with_pos)

#         # Pass through the main transformer blocks
#         encoded_tokens = self.encoder_model.transformer(x_dropped_out)

#         # Pool the output of the transformer to get a single latent vector
#         if self.encoder_model.pool == 'cls':
#             latent_representation = encoded_tokens[:, 0]
#         elif self.encoder_model.pool == 'mean':
#             latent_representation = encoded_tokens.mean(dim=1)
#         else:
#             raise ValueError(f"Unknown pooling type: {self.encoder_model.pool}")

#         # latent: (batch, latent_dim)
#         # Want: (batch, N, latent_dim)
#         N = h * w
#         latent_seq = latent_representation.unsqueeze(1).repeat(1, N, 1)

#         return latent_seq

# Path to your buffer file
buffer_path = "data/buffer.h5"

# Open the buffer and extract the state
with h5py.File(buffer_path, "r") as f:
    state_np = f["state"][:]  # Load the entire state dataset into a numpy array

print("State shape:", state_np.shape)
print("State dtype:", state_np.dtype)

# Convert to torch tensor
state_tensor = torch.from_numpy(state_np)