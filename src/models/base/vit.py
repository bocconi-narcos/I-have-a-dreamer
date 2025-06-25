from einops import repeat
import torch
import torch.nn as nn
from src.utils.weight_init import initialize_weights
from einops import rearrange
from einops.layers.torch import Rearrange
from .transformer_blocks import PreNorm, FeedForward, Attention, Transformer

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * \
            (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        ) if num_classes > 0 else nn.Identity()  # Only add mlp_head if num_classes is positive

        self.apply(initialize_weights)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        # Decide how to pool features
        if self.pool == 'mean':
            x = x.mean(dim=1)
        else:  # 'cls'
            x = x[:, 0]  # Take the CLS token

        latent_representation = self.to_latent(x)

        if self.mlp_head is nn.Identity():
            return latent_representation
        else:
            return self.mlp_head(latent_representation)

# Need to add 'repeat' for the cls_token. It's often part of einops.
# If it's not directly in the main einops, it might be in einops.layers or a separate import.
# Let's assume 'repeat' is available from einops for now.
# from einops import repeat # This should be at the top if not already there.
# It appears 'repeat' is not automatically imported with 'from einops import rearrange'.
# Let's ensure it's imported.


# Correcting imports for 'repeat'

# Final check on ViT class structure:
# The ViT class should take image_size, patch_size, dim (output latent dim), depth, heads, mlp_dim.
# The 'num_classes' can be set to the latent_dim if we want the mlp_head to project to that,
# or 0 if we want the raw pooled output.
# For our case, we often want the raw latent vector, so perhaps 'num_classes=0' is a good default for that.
# The current implementation has `mlp_head` which would be an identity if `num_classes=0`.
# This means it will return `latent_representation` as desired when `num_classes=0`.
