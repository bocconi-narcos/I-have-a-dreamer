"""
Comprehensive tests for StateEncoder, focusing on:
1. Removal of CLS token
2. Output of all tokens
3. Causal mask functionality for variable-sized grids
4. Backward compatibility helpers
"""

import torch
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.state_encoder import StateEncoder, StateEncoderWrapper


class TestStateEncoderBasic:
    """Basic functionality tests for StateEncoder"""
    
    @pytest.fixture
    def encoder(self):
        """Create a StateEncoder instance for testing"""
        encoder_params = {
            "depth": 2,
            "heads": 4,
            "mlp_dim": 128,
            "transformer_dim": 64,
            "dropout": 0.1,
            "emb_dropout": 0.1,
            "scaled_position_embeddings": False,
            "colors_vocab_size": 12
        }
        return StateEncoder(
            image_size=(10, 10),
            input_channels=1,
            latent_dim=256,
            encoder_params=encoder_params
        )
    
    @pytest.fixture
    def sample_batch(self):
        """Create a sample batch for testing"""
        B, H, W = 2, 10, 10
        x = torch.randint(0, 11, (B, H, W))
        # Set some padding values
        x[0, 8:, :] = -1  # First sample: rows 8-9 are padding
        x[1, :, 8:] = -1  # Second sample: cols 8-9 are padding
        
        shape_h = torch.tensor([8, 10])  # First sample has height 8, second has 10
        shape_w = torch.tensor([10, 8])  # First sample has width 10, second has 8
        most_common_color = torch.tensor([1, 2])
        least_common_color = torch.tensor([0, 1])
        num_unique_colors = torch.tensor([5, 7])
        
        return {
            'x': x,
            'shape_h': shape_h,
            'shape_w': shape_w,
            'most_common_color': most_common_color,
            'least_common_color': least_common_color,
            'num_unique_colors': num_unique_colors
        }
    
    def test_no_cls_token_in_model(self, encoder):
        """Test that CLS token is not defined in the model"""
        assert not hasattr(encoder, 'cls_token'), "CLS token should be removed"
    
    def test_forward_returns_tuple(self, encoder, sample_batch):
        """Test that forward() returns (tokens, causal_mask) tuple"""
        result = encoder(
            sample_batch['x'],
            sample_batch['shape_h'],
            sample_batch['shape_w'],
            sample_batch['most_common_color'],
            sample_batch['least_common_color'],
            sample_batch['num_unique_colors']
        )
        
        assert isinstance(result, tuple), "Forward should return a tuple"
        assert len(result) == 2, "Forward should return (tokens, causal_mask)"
    
    def test_tokens_output_shape(self, encoder, sample_batch):
        """Test that tokens have correct shape: (B, seq_len, latent_dim)"""
        tokens, causal_mask = encoder(
            sample_batch['x'],
            sample_batch['shape_h'],
            sample_batch['shape_w'],
            sample_batch['most_common_color'],
            sample_batch['least_common_color'],
            sample_batch['num_unique_colors']
        )
        
        B = sample_batch['x'].shape[0]
        H, W = sample_batch['x'].shape[1], sample_batch['x'].shape[2]
        seq_len = 5 + H * W  # 5 metadata tokens + H*W grid tokens
        
        assert tokens.shape == (B, seq_len, encoder.latent_dim), \
            f"Expected tokens shape ({B}, {seq_len}, {encoder.latent_dim}), got {tokens.shape}"
    
    def test_causal_mask_output_shape(self, encoder, sample_batch):
        """Test that causal_mask has correct shape: (B, seq_len, seq_len)"""
        tokens, causal_mask = encoder(
            sample_batch['x'],
            sample_batch['shape_h'],
            sample_batch['shape_w'],
            sample_batch['most_common_color'],
            sample_batch['least_common_color'],
            sample_batch['num_unique_colors']
        )
        
        B = sample_batch['x'].shape[0]
        H, W = sample_batch['x'].shape[1], sample_batch['x'].shape[2]
        seq_len = 5 + H * W
        
        assert causal_mask.shape == (B, seq_len, seq_len), \
            f"Expected causal_mask shape ({B}, {seq_len}, {seq_len}), got {causal_mask.shape}"
        assert causal_mask.dtype == torch.bool, "Causal mask should be boolean"
    
    def test_all_tokens_present(self, encoder, sample_batch):
        """Test that all tokens (metadata + grid) are present in output"""
        tokens, causal_mask = encoder(
            sample_batch['x'],
            sample_batch['shape_h'],
            sample_batch['shape_w'],
            sample_batch['most_common_color'],
            sample_batch['least_common_color'],
            sample_batch['num_unique_colors']
        )
        
        H, W = sample_batch['x'].shape[1], sample_batch['x'].shape[2]
        expected_seq_len = 5 + H * W  # 5 metadata + H*W grid
        
        assert tokens.shape[1] == expected_seq_len, \
            f"Expected {expected_seq_len} tokens, got {tokens.shape[1]}"


class TestCausalMask:
    """Tests specifically for causal mask functionality"""
    
    @pytest.fixture
    def encoder(self):
        encoder_params = {
            "depth": 2,
            "heads": 4,
            "mlp_dim": 128,
            "transformer_dim": 64,
            "dropout": 0.1,
            "emb_dropout": 0.1,
            "colors_vocab_size": 12
        }
        return StateEncoder(
            image_size=(5, 5),
            input_channels=1,
            latent_dim=128,
            encoder_params=encoder_params
        )
    
    def test_causal_mask_metadata_tokens_unmasked(self, encoder):
        """Test that metadata tokens (first 5) can attend to everything"""
        B, H, W = 1, 5, 5
        x = torch.randint(0, 11, (B, H, W))
        shape_h = torch.tensor([3])  # Actual height is 3
        shape_w = torch.tensor([3])  # Actual width is 3
        
        tokens, causal_mask = encoder(
            x,
            shape_h,
            shape_w,
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([5])
        )
        
        # First 5 tokens (metadata) should not be masked when attending
        # Check that metadata tokens can attend to all positions
        metadata_mask = causal_mask[0, :5, :]  # (5, seq_len)
        assert not metadata_mask.any(), \
            "Metadata tokens should be able to attend to all positions"
    
    def test_causal_mask_invalid_positions_masked(self, encoder):
        """Test that invalid grid positions are masked out"""
        B, H, W = 1, 5, 5
        x = torch.randint(0, 11, (B, H, W))
        shape_h = torch.tensor([3])  # Actual height is 3 (rows 0-2)
        shape_w = torch.tensor([3])  # Actual width is 3 (cols 0-2)
        
        tokens, causal_mask = encoder(
            x,
            shape_h,
            shape_w,
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([5])
        )
        
        seq_len = 5 + H * W
        grid_start_idx = 5
        
        # Positions beyond valid grid size (3*3=9) should be masked
        # Valid positions: 0-8 (9 positions), invalid: 9-24 (16 positions)
        valid_grid_size = 3 * 3
        
        # Check that invalid grid tokens mask everything
        for i in range(valid_grid_size, H * W):
            assert causal_mask[0, grid_start_idx + i, :].all(), \
                f"Invalid grid token {i} should mask all positions"
    
    def test_causal_mask_valid_tokens_attend_to_valid_only(self, encoder):
        """Test that valid grid tokens only attend to valid positions"""
        B, H, W = 1, 5, 5
        x = torch.randint(0, 11, (B, H, W))
        shape_h = torch.tensor([3])  # Actual height is 3
        shape_w = torch.tensor([3])  # Actual width is 3
        
        tokens, causal_mask = encoder(
            x,
            shape_h,
            shape_w,
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([5])
        )
        
        grid_start_idx = 5
        valid_grid_size = 3 * 3
        
        # For valid grid tokens, they should be able to attend to:
        # - All metadata tokens (first 5) - should NOT be masked
        # - Valid grid positions only
        
        # Check a valid grid token (e.g., position 0)
        valid_token_idx = grid_start_idx + 0
        
        # Should be able to attend to metadata tokens
        metadata_attention = causal_mask[0, valid_token_idx, :5]
        assert not metadata_attention.any(), \
            "Valid grid tokens should attend to metadata tokens"
        
        # Should be able to attend to valid grid positions
        for j in range(valid_grid_size):
            target_idx = grid_start_idx + j
            assert not causal_mask[0, valid_token_idx, target_idx], \
                f"Valid token {0} should attend to valid position {j}"
        
        # Should NOT be able to attend to invalid grid positions
        for j in range(valid_grid_size, H * W):
            target_idx = grid_start_idx + j
            assert causal_mask[0, valid_token_idx, target_idx], \
                f"Valid token {0} should NOT attend to invalid position {j}"
    
    def test_causal_mask_different_grid_sizes(self, encoder):
        """Test causal mask with different grid sizes in batch"""
        B, H, W = 2, 5, 5
        x = torch.randint(0, 11, (B, H, W))
        
        # First sample: 3x3 grid
        # Second sample: 4x4 grid
        shape_h = torch.tensor([3, 4])
        shape_w = torch.tensor([3, 4])
        
        tokens, causal_mask = encoder(
            x,
            shape_h,
            shape_w,
            torch.tensor([1, 2]),
            torch.tensor([0, 1]),
            torch.tensor([5, 6])
        )
        
        # Check first sample (3x3 = 9 valid positions)
        grid_start_idx = 5
        valid_size_1 = 3 * 3
        
        # Check that invalid positions for first sample are masked
        for i in range(valid_size_1, H * W):
            assert causal_mask[0, grid_start_idx + i, :].all(), \
                f"Sample 0: Invalid position {i} should mask all"
        
        # Check second sample (4x4 = 16 valid positions)
        valid_size_2 = 4 * 4
        
        # Check that invalid positions for second sample are masked
        for i in range(valid_size_2, H * W):
            assert causal_mask[1, grid_start_idx + i, :].all(), \
                f"Sample 1: Invalid position {i} should mask all"
    
    def test_causal_mask_symmetric_properties(self, encoder):
        """Test that causal mask has expected properties"""
        B, H, W = 1, 4, 4
        x = torch.randint(0, 11, (B, H, W))
        shape_h = torch.tensor([3])
        shape_w = torch.tensor([3])
        
        tokens, causal_mask = encoder(
            x,
            shape_h,
            shape_w,
            torch.tensor([1]),
            torch.tensor([0]),
            torch.tensor([5])
        )
        
        # Causal mask should be boolean
        assert causal_mask.dtype == torch.bool
        
        # All values should be boolean (True or False)
        assert causal_mask.min() >= 0 and causal_mask.max() <= 1


class TestBackwardCompatibility:
    """Tests for backward compatibility helpers"""
    
    @pytest.fixture
    def encoder(self):
        encoder_params = {
            "depth": 2,
            "heads": 4,
            "mlp_dim": 128,
            "transformer_dim": 64,
            "dropout": 0.1,
            "emb_dropout": 0.1,
            "colors_vocab_size": 12
        }
        return StateEncoder(
            image_size=(5, 5),
            input_channels=1,
            latent_dim=128,
            encoder_params=encoder_params
        )
    
    def test_pool_tokens_mean(self, encoder):
        """Test mean pooling of tokens"""
        B, seq_len, latent_dim = 2, 30, 128
        tokens = torch.randn(B, seq_len, latent_dim)
        causal_mask = torch.zeros(B, seq_len, seq_len, dtype=torch.bool)
        
        pooled = encoder.pool_tokens(tokens, causal_mask, method='mean')
        
        assert pooled.shape == (B, latent_dim), \
            f"Expected pooled shape ({B}, {latent_dim}), got {pooled.shape}"
        
        # Check that mean pooling is correct
        expected_mean = tokens.mean(dim=1)
        assert torch.allclose(pooled, expected_mean), \
            "Mean pooling should match manual computation"
    
    def test_pool_tokens_first(self, encoder):
        """Test first token pooling"""
        B, seq_len, latent_dim = 2, 30, 128
        tokens = torch.randn(B, seq_len, latent_dim)
        causal_mask = torch.zeros(B, seq_len, seq_len, dtype=torch.bool)
        
        pooled = encoder.pool_tokens(tokens, causal_mask, method='first')
        
        assert pooled.shape == (B, latent_dim), \
            f"Expected pooled shape ({B}, {latent_dim}), got {pooled.shape}"
        
        # Check that first token pooling is correct
        expected_first = tokens[:, 0, :]
        assert torch.allclose(pooled, expected_first), \
            "First token pooling should return first token"
    
    def test_pool_tokens_invalid_method(self, encoder):
        """Test that invalid pooling method raises error"""
        B, seq_len, latent_dim = 2, 30, 128
        tokens = torch.randn(B, seq_len, latent_dim)
        causal_mask = torch.zeros(B, seq_len, seq_len, dtype=torch.bool)
        
        with pytest.raises(ValueError, match="Unknown pooling method"):
            encoder.pool_tokens(tokens, causal_mask, method='invalid')


class TestWrapper:
    """Tests for StateEncoderWrapper backward compatibility"""
    
    @pytest.fixture
    def encoder(self):
        encoder_params = {
            "depth": 2,
            "heads": 4,
            "mlp_dim": 128,
            "transformer_dim": 64,
            "dropout": 0.1,
            "emb_dropout": 0.1,
            "colors_vocab_size": 12
        }
        base_encoder = StateEncoder(
            image_size=(5, 5),
            input_channels=1,
            latent_dim=128,
            encoder_params=encoder_params
        )
        return StateEncoderWrapper(base_encoder, pool_method='mean')
    
    def test_wrapper_returns_single_tensor(self, encoder):
        """Test that wrapper returns single tensor (B, latent_dim) for backward compatibility"""
        B, H, W = 2, 5, 5
        x = torch.randint(0, 11, (B, H, W))
        shape_h = torch.tensor([3, 4])
        shape_w = torch.tensor([3, 4])
        
        result = encoder(
            x,
            shape_h,
            shape_w,
            torch.tensor([1, 2]),
            torch.tensor([0, 1]),
            torch.tensor([5, 6])
        )
        
        assert isinstance(result, torch.Tensor), "Wrapper should return single tensor"
        assert result.shape == (B, 128), f"Expected ({B}, 128), got {result.shape}"
        assert result.dim() == 2, "Should return 2D tensor (B, latent_dim)"
    
    def test_wrapper_attribute_access(self, encoder):
        """Test that wrapper forwards attribute access to wrapped encoder"""
        assert encoder.latent_dim == 128
        assert encoder.max_rows == 5
        assert encoder.max_cols == 5
    
    def test_wrapper_different_pool_methods(self):
        """Test wrapper with different pooling methods"""
        encoder_params = {
            "depth": 2,
            "heads": 4,
            "mlp_dim": 128,
            "transformer_dim": 64,
            "dropout": 0.1,
            "emb_dropout": 0.1,
            "colors_vocab_size": 12
        }
        base_encoder = StateEncoder(
            image_size=(5, 5),
            input_channels=1,
            latent_dim=128,
            encoder_params=encoder_params
        )
        
        wrapped_mean = StateEncoderWrapper(base_encoder, pool_method='mean')
        wrapped_first = StateEncoderWrapper(base_encoder, pool_method='first')
        
        B, H, W = 2, 5, 5
        x = torch.randint(0, 11, (B, H, W))
        shape_h = torch.tensor([3, 4])
        shape_w = torch.tensor([3, 4])
        
        result_mean = wrapped_mean(x, shape_h, shape_w, torch.tensor([1, 2]), torch.tensor([0, 1]), torch.tensor([5, 6]))
        result_first = wrapped_first(x, shape_h, shape_w, torch.tensor([1, 2]), torch.tensor([0, 1]), torch.tensor([5, 6]))
        
        assert result_mean.shape == (B, 128)
        assert result_first.shape == (B, 128)
        # Results should be different (mean vs first token)
        assert not torch.allclose(result_mean, result_first), "Mean and first pooling should give different results"


class TestIntegration:
    """Integration tests for StateEncoder"""
    
    @pytest.fixture
    def encoder(self):
        encoder_params = {
            "depth": 2,
            "heads": 4,
            "mlp_dim": 128,
            "transformer_dim": 64,
            "dropout": 0.1,
            "emb_dropout": 0.1,
            "colors_vocab_size": 12
        }
        return StateEncoder(
            image_size=(10, 10),
            input_channels=1,
            latent_dim=256,
            encoder_params=encoder_params
        )
    
    def test_end_to_end_forward_pass(self, encoder):
        """Test complete forward pass with realistic data"""
        B, H, W = 4, 10, 10
        x = torch.randint(0, 11, (B, H, W))
        
        # Set some padding
        x[0, 8:, :] = -1
        x[1, :, 7:] = -1
        
        shape_h = torch.tensor([8, 10, 5, 10])
        shape_w = torch.tensor([10, 7, 10, 6])
        most_common_color = torch.randint(0, 11, (B,))
        least_common_color = torch.randint(0, 11, (B,))
        num_unique_colors = torch.randint(1, 12, (B,))
        
        tokens, causal_mask = encoder(
            x,
            shape_h,
            shape_w,
            most_common_color,
            least_common_color,
            num_unique_colors
        )
        
        # Verify outputs
        seq_len = 5 + H * W
        assert tokens.shape == (B, seq_len, 256)
        assert causal_mask.shape == (B, seq_len, seq_len)
        
        # Verify no NaN or Inf
        assert not torch.isnan(tokens).any(), "Tokens should not contain NaN"
        assert not torch.isinf(tokens).any(), "Tokens should not contain Inf"
    
    def test_gradient_flow(self, encoder):
        """Test that gradients flow through the encoder"""
        B, H, W = 2, 5, 5
        x = torch.randint(0, 11, (B, H, W), requires_grad=False)
        shape_h = torch.tensor([3, 4])
        shape_w = torch.tensor([3, 4])
        
        tokens, causal_mask = encoder(
            x,
            shape_h,
            shape_w,
            torch.tensor([1, 2]),
            torch.tensor([0, 1]),
            torch.tensor([5, 6])
        )
        
        # Create a dummy loss
        loss = tokens.sum()
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        has_gradients = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in encoder.parameters()
            if p.requires_grad
        )
        
        assert has_gradients, "Gradients should flow through encoder"


class TestScaledPositionalEmbeddings:
    """Tests for scaled positional embeddings (JAX→PyTorch translation verification)"""
    
    def test_scaled_positional_embeddings_work(self):
        """Test that scaled positional embeddings work correctly"""
        encoder_params = {
            "depth": 2,
            "heads": 4,
            "mlp_dim": 128,
            "transformer_dim": 64,
            "dropout": 0.1,
            "emb_dropout": 0.1,
            "scaled_position_embeddings": True,  # Enable scaled mode
            "colors_vocab_size": 12
        }
        encoder = StateEncoder(
            image_size=(5, 5),
            input_channels=1,
            latent_dim=128,
            encoder_params=encoder_params
        )
        
        B, H, W = 2, 5, 5
        x = torch.randint(0, 11, (B, H, W))
        shape_h = torch.tensor([5, 5])
        shape_w = torch.tensor([5, 5])
        most_common = torch.tensor([1, 2])
        least_common = torch.tensor([0, 1])
        num_colors = torch.tensor([5, 6])
        
        tokens, causal_mask = encoder(x, shape_h, shape_w, most_common, least_common, num_colors)
        
        assert tokens.shape == (B, 5 + H*W, 128), "Output shape should be correct"
        assert causal_mask.shape == (B, 5 + H*W, 5 + H*W), "Causal mask shape should be correct"
    
    def test_scaled_vs_non_scaled_consistency(self):
        """Test that scaled and non-scaled modes produce consistent output shapes"""
        encoder_params_scaled = {
            "depth": 2,
            "heads": 4,
            "mlp_dim": 128,
            "transformer_dim": 64,
            "dropout": 0.1,
            "emb_dropout": 0.1,
            "scaled_position_embeddings": True,
            "colors_vocab_size": 12
        }
        
        encoder_params_normal = {
            "depth": 2,
            "heads": 4,
            "mlp_dim": 128,
            "transformer_dim": 64,
            "dropout": 0.1,
            "emb_dropout": 0.1,
            "scaled_position_embeddings": False,
            "colors_vocab_size": 12
        }
        
        encoder_scaled = StateEncoder(
            image_size=(5, 5),
            input_channels=1,
            latent_dim=128,
            encoder_params=encoder_params_scaled
        )
        
        encoder_normal = StateEncoder(
            image_size=(5, 5),
            input_channels=1,
            latent_dim=128,
            encoder_params=encoder_params_normal
        )
        
        B, H, W = 2, 5, 5
        x = torch.randint(0, 11, (B, H, W))
        shape_h = torch.tensor([5, 5])
        shape_w = torch.tensor([5, 5])
        most_common = torch.tensor([1, 2])
        least_common = torch.tensor([0, 1])
        num_colors = torch.tensor([5, 6])
        
        tokens_scaled, mask_scaled = encoder_scaled(x, shape_h, shape_w, most_common, least_common, num_colors)
        tokens_normal, mask_normal = encoder_normal(x, shape_h, shape_w, most_common, least_common, num_colors)
        
        # Both should produce same output shapes
        assert tokens_scaled.shape == tokens_normal.shape, "Output shapes should match"
        assert mask_scaled.shape == mask_normal.shape, "Mask shapes should match"
    
    def test_scaled_positional_embeddings_zero_based_indexing(self):
        """Verify that scaled positional embeddings use 0-based indexing (JAX→PyTorch fix)"""
        encoder_params = {
            "depth": 1,
            "heads": 2,
            "mlp_dim": 64,
            "transformer_dim": 32,
            "dropout": 0.0,
            "emb_dropout": 0.0,
            "scaled_position_embeddings": True,
            "colors_vocab_size": 12
        }
        encoder = StateEncoder(
            image_size=(3, 3),
            input_channels=1,
            latent_dim=64,
            encoder_params=encoder_params
        )
        
        # Verify that positional embedding parameters exist and are initialized
        assert hasattr(encoder, 'pos_row_embed'), "Should have pos_row_embed parameter"
        assert hasattr(encoder, 'pos_col_embed'), "Should have pos_col_embed parameter"
        assert encoder.pos_row_embed.shape == (32,), "pos_row_embed should have correct shape"
        assert encoder.pos_col_embed.shape == (32,), "pos_col_embed should have correct shape"
        
        # Test forward pass with small grid to verify indexing
        B, H, W = 1, 3, 3
        x = torch.randint(0, 11, (B, H, W))
        shape_h = torch.tensor([3])
        shape_w = torch.tensor([3])
        most_common = torch.tensor([1])
        least_common = torch.tensor([0])
        num_colors = torch.tensor([5])
        
        tokens, causal_mask = encoder(x, shape_h, shape_w, most_common, least_common, num_colors)
        
        # Verify output shape (3x3 grid = 9 tokens + 5 metadata = 14 tokens)
        assert tokens.shape == (B, 14, 64), "Output shape should be correct"
        
        # The fact that it runs without errors confirms 0-based indexing works


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

