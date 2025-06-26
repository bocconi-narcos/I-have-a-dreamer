import torch
import torch.nn as nn

class ActionEmbedder(nn.Module):
    """
    Generic embedding layer for one-hot encoded actions or action indices,
    with optional dropout on the embeddings.

    This module supports two types of inputs:
      - Integer indices of shape (batch_size,) or (batch_size, seq_len)
      - One-hot encoded vectors of shape (batch_size, num_actions) or (batch_size, seq_len, num_actions)

    Args:
        num_actions (int): Total number of distinct actions.
        embed_dim (int): Dimension of the embedding space.
        dropout_p (float): Dropout probability applied to embeddings (default: 0.0).

    Usage:
        embedder = ActionEmbedder(num_actions=10, embed_dim=32, dropout_p=0.1)
        idx = torch.LongTensor([0, 3, 5])
        out = embedder(idx)

        one_hot = torch.eye(10)[idx]
        out = embedder(one_hot)
    """
    def __init__(self, num_actions: int, embed_dim: int, dropout_p: float = 0.0):
        super().__init__()
        self.num_actions = num_actions
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p

        # learnable embedding matrix
        self.weight = nn.Parameter(torch.Tensor(num_actions, embed_dim))
        # dropout layer
        self.dropout = nn.Dropout(p=dropout_p)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize embedding weights with a normal distribution."""
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / (self.embed_dim ** 0.5))

    def forward(self, actions):
        """
        Forward pass to embed actions and apply dropout.

        Args:
            actions (Tensor): Either LongTensor of indices or FloatTensor one-hot vectors.
        Returns:
            Tensor: Embedded representations of shape (..., embed_dim)
        """
        # Compute embeddings
        if actions.dtype in (torch.long, torch.int):
            embed = torch.embedding(self.weight, actions)
        else:
            # assume float one-hot vectors
            original_shape = actions.shape
            # reshape sequences to 2D if necessary
            if actions.dim() > 2:
                flat_batch = int(torch.prod(torch.tensor(original_shape[:-1])))
                one_hot_flat = actions.view(flat_batch, self.num_actions)
                embed_flat = one_hot_flat.matmul(self.weight)
                embed = embed_flat.view(*original_shape[:-1], self.embed_dim)
            else:
                embed = actions.matmul(self.weight)

        # Apply dropout
        return self.dropout(embed)

if __name__ == "__main__":
    # simple sanity check
    embedder = ActionEmbedder(num_actions=5, embed_dim=8, dropout_p=0.2)
    idx = torch.LongTensor([0, 2, 4])
    print('Indices input:', embedder(idx).shape)
    oh = torch.eye(5)[idx]
    print('One-hot input:', embedder(oh).shape)
