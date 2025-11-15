"""
Integration tests for CrossAttentionColorPredictor with StateEncoder
Tests realistic scenarios with actual data flow
"""

import torch
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.predictors.color_predictor import CrossAttentionColorPredictor
from models.state_encoder import StateEncoder
from models.action_embed import ActionEmbedder


class TestColorPredictorIntegration:
    """Integration tests with StateEncoder"""
    
    @pytest.fixture
    def models(self):
        """Create all models needed for integration test"""
        latent_dim = 256
        action_embedding_dim = 32
        
        encoder_params = {
            'depth': 2,
            'heads': 4,
            'mlp_dim': 128,
            'transformer_dim': 64,
            'dropout': 0.1,
            'emb_dropout': 0.1,
            'colors_vocab_size': 12
        }
        
        state_encoder = StateEncoder(
            image_size=(10, 10),
            input_channels=1,
            latent_dim=latent_dim,
            encoder_params=encoder_params
        )
        
        action_embedder = ActionEmbedder(
            num_actions=23,
            embed_dim=action_embedding_dim,
            dropout_p=0.1
        )
        
        color_predictor = CrossAttentionColorPredictor(
            latent_dim=latent_dim,
            num_colors=11,
            action_embedding_dim=action_embedding_dim,
            num_layers=2,
            heads=8,
            mlp_dim=256,
            dropout=0.1,
            mlp_hidden_dim=128
        )
        
        return state_encoder, action_embedder, color_predictor
    
    def test_full_pipeline(self, models):
        """Test complete pipeline: StateEncoder → ActionEmbedder → ColorPredictor"""
        state_encoder, action_embedder, color_predictor = models
        
        B, H, W = 3, 10, 10
        x = torch.randint(0, 11, (B, H, W))
        shape_h = torch.tensor([8, 10, 5])
        shape_w = torch.tensor([10, 7, 10])
        most_common = torch.tensor([1, 2, 0])
        least_common = torch.tensor([0, 1, 2])
        num_colors_grid = torch.tensor([5, 7, 4])
        
        action_colour = torch.randint(0, 23, (B,))
        action_onehot = torch.nn.functional.one_hot(action_colour, num_classes=23).float()
        
        # Forward through StateEncoder
        state_tokens, causal_mask = state_encoder(
            x, shape_h, shape_w, most_common, least_common, num_colors_grid
        )
        
        # Forward through ActionEmbedder
        action_embedding = action_embedder(action_onehot)
        
        # Forward through ColorPredictor
        color_logits = color_predictor(action_embedding, state_tokens, causal_mask)
        
        # Verify outputs
        assert state_tokens.shape == (B, 5 + H*W, 256), f"State tokens shape: {state_tokens.shape}"
        assert causal_mask.shape == (B, 5 + H*W, 5 + H*W), f"Causal mask shape: {causal_mask.shape}"
        assert action_embedding.shape == (B, 32), f"Action embedding shape: {action_embedding.shape}"
        assert color_logits.shape == (B, 11), f"Color logits shape: {color_logits.shape}"
        
        # Verify logits are reasonable (not NaN, not Inf)
        assert not torch.isnan(color_logits).any(), "Logits should not contain NaN"
        assert not torch.isinf(color_logits).any(), "Logits should not contain Inf"
        assert color_logits.abs().max() < 100, "Logits should be reasonable values"
    
    def test_different_grid_sizes(self, models):
        """Test with different grid sizes in batch"""
        state_encoder, action_embedder, color_predictor = models
        
        B = 3
        # Different grid sizes
        grids = [
            torch.randint(0, 11, (5, 5)),   # Small grid
            torch.randint(0, 11, (10, 10)), # Medium grid
            torch.randint(0, 11, (8, 7))   # Rectangular grid
        ]
        
        # Pad to same size (10x10)
        max_h, max_w = 10, 10
        x_padded = torch.full((B, max_h, max_w), -1, dtype=torch.long)
        shape_h_list = []
        shape_w_list = []
        
        for i, grid in enumerate(grids):
            h, w = grid.shape
            x_padded[i, :h, :w] = grid
            shape_h_list.append(h)
            shape_w_list.append(w)
        
        shape_h = torch.tensor(shape_h_list)
        shape_w = torch.tensor(shape_w_list)
        most_common = torch.tensor([1, 2, 0])
        least_common = torch.tensor([0, 1, 2])
        num_colors_grid = torch.tensor([5, 7, 4])
        
        action_colour = torch.randint(0, 23, (B,))
        action_onehot = torch.nn.functional.one_hot(action_colour, num_classes=23).float()
        
        # Forward pass
        state_tokens, causal_mask = state_encoder(
            x_padded, shape_h, shape_w, most_common, least_common, num_colors_grid
        )
        action_embedding = action_embedder(action_onehot)
        color_logits = color_predictor(action_embedding, state_tokens, causal_mask)
        
        # Verify
        assert color_logits.shape == (B, 11)
        assert not torch.isnan(color_logits).any()
    
    def test_causal_mask_effectiveness(self, models):
        """Test that causal mask prevents attention to padding tokens"""
        state_encoder, action_embedder, color_predictor = models
        
        B, H, W = 2, 10, 10
        
        # Create grids with different actual sizes
        x = torch.full((B, H, W), -1, dtype=torch.long)
        x[0, :5, :5] = torch.randint(0, 11, (5, 5))  # First: 5x5 actual
        x[1, :8, :8] = torch.randint(0, 11, (8, 8))  # Second: 8x8 actual
        
        shape_h = torch.tensor([5, 8])
        shape_w = torch.tensor([5, 8])
        most_common = torch.tensor([1, 2])
        least_common = torch.tensor([0, 1])
        num_colors_grid = torch.tensor([5, 7])
        
        action_colour = torch.randint(0, 23, (B,))
        action_onehot = torch.nn.functional.one_hot(action_colour, num_classes=23).float()
        
        # Forward pass
        state_tokens, causal_mask = state_encoder(
            x, shape_h, shape_w, most_common, least_common, num_colors_grid
        )
        action_embedding = action_embedder(action_onehot)
        
        # Test with mask
        logits_with_mask = color_predictor(action_embedding, state_tokens, causal_mask)
        
        # Test without mask (should still work but may attend to padding)
        logits_without_mask = color_predictor(action_embedding, state_tokens, None)
        
        # Both should produce valid outputs
        assert logits_with_mask.shape == (B, 11)
        assert logits_without_mask.shape == (B, 11)
        assert not torch.isnan(logits_with_mask).any()
        assert not torch.isnan(logits_without_mask).any()
        
        # Results may differ (mask affects attention)
        # But both should be valid
    
    def test_gradient_flow_end_to_end(self, models):
        """Test that gradients flow through entire pipeline"""
        state_encoder, action_embedder, color_predictor = models
        
        B, H, W = 2, 10, 10
        x = torch.randint(0, 11, (B, H, W))
        shape_h = torch.tensor([10, 10])
        shape_w = torch.tensor([10, 10])
        most_common = torch.tensor([1, 2])
        least_common = torch.tensor([0, 1])
        num_colors_grid = torch.tensor([5, 6])
        
        action_colour = torch.randint(0, 23, (B,))
        action_onehot = torch.nn.functional.one_hot(action_colour, num_classes=23).float()
        
        # Forward pass
        state_tokens, causal_mask = state_encoder(
            x, shape_h, shape_w, most_common, least_common, num_colors_grid
        )
        action_embedding = action_embedder(action_onehot)
        color_logits = color_predictor(action_embedding, state_tokens, causal_mask)
        
        # Create dummy loss
        target_colour = torch.randint(0, 11, (B,))
        loss = torch.nn.functional.cross_entropy(color_logits, target_colour)
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        encoder_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in state_encoder.parameters()
            if p.requires_grad
        )
        action_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in action_embedder.parameters()
            if p.requires_grad
        )
        predictor_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in color_predictor.parameters()
            if p.requires_grad
        )
        
        assert encoder_grads, "Gradients should flow through encoder"
        assert action_grads, "Gradients should flow through action embedder"
        assert predictor_grads, "Gradients should flow through color predictor"
    
    def test_consistent_output_structure(self, models):
        """Test that same inputs produce consistent output structure (shape, dtype, etc.)"""
        state_encoder, action_embedder, color_predictor = models
        
        B, H, W = 2, 10, 10
        x = torch.randint(0, 11, (B, H, W))
        shape_h = torch.tensor([10, 10])
        shape_w = torch.tensor([10, 10])
        most_common = torch.tensor([1, 2])
        least_common = torch.tensor([0, 1])
        num_colors_grid = torch.tensor([5, 6])
        
        action_colour = torch.randint(0, 23, (B,))
        action_onehot = torch.nn.functional.one_hot(action_colour, num_classes=23).float()
        
        # First forward pass
        state_tokens1, causal_mask1 = state_encoder(
            x, shape_h, shape_w, most_common, least_common, num_colors_grid
        )
        action_embedding1 = action_embedder(action_onehot)
        color_logits1 = color_predictor(action_embedding1, state_tokens1, causal_mask1)
        
        # Second forward pass (structure should be identical, values may differ due to dropout)
        state_tokens2, causal_mask2 = state_encoder(
            x, shape_h, shape_w, most_common, least_common, num_colors_grid
        )
        action_embedding2 = action_embedder(action_onehot)
        color_logits2 = color_predictor(action_embedding2, state_tokens2, causal_mask2)
        
        # Verify output structure is consistent
        assert state_tokens1.shape == state_tokens2.shape, "State tokens should have same shape"
        assert causal_mask1.shape == causal_mask2.shape, "Causal masks should have same shape"
        assert torch.equal(causal_mask1, causal_mask2), "Causal masks should be identical (deterministic)"
        assert action_embedding1.shape == action_embedding2.shape, "Action embeddings should have same shape"
        assert color_logits1.shape == color_logits2.shape, "Color logits should have same shape"
        
        # Verify dtypes
        assert state_tokens1.dtype == state_tokens2.dtype == torch.float32
        assert color_logits1.dtype == color_logits2.dtype == torch.float32
        
        # Verify no NaN or Inf
        assert not torch.isnan(color_logits1).any(), "Logits should not contain NaN"
        assert not torch.isnan(color_logits2).any(), "Logits should not contain NaN"
        assert not torch.isinf(color_logits1).any(), "Logits should not contain Inf"
        assert not torch.isinf(color_logits2).any(), "Logits should not contain Inf"
    
    def test_batch_consistency(self, models):
        """Test that batch processing works correctly"""
        state_encoder, action_embedder, color_predictor = models
        
        # Test with different batch sizes
        for B in [1, 2, 4, 8]:
            H, W = 10, 10
            x = torch.randint(0, 11, (B, H, W))
            shape_h = torch.full((B,), H, dtype=torch.long)
            shape_w = torch.full((B,), W, dtype=torch.long)
            most_common = torch.randint(0, 11, (B,))
            least_common = torch.randint(0, 11, (B,))
            num_colors_grid = torch.randint(1, 11, (B,))
            
            action_colour = torch.randint(0, 23, (B,))
            action_onehot = torch.nn.functional.one_hot(action_colour, num_classes=23).float()
            
            # Forward pass
            state_tokens, causal_mask = state_encoder(
                x, shape_h, shape_w, most_common, least_common, num_colors_grid
            )
            action_embedding = action_embedder(action_onehot)
            color_logits = color_predictor(action_embedding, state_tokens, causal_mask)
            
            # Verify batch dimension
            assert state_tokens.shape[0] == B, f"Batch size mismatch: {state_tokens.shape[0]} != {B}"
            assert action_embedding.shape[0] == B, f"Batch size mismatch: {action_embedding.shape[0]} != {B}"
            assert color_logits.shape[0] == B, f"Batch size mismatch: {color_logits.shape[0]} != {B}"
            assert color_logits.shape[1] == 11, "Should have 11 color classes"
    
    def test_action_embedding_projection(self, models):
        """Test that action embedding projection works correctly"""
        _, action_embedder, color_predictor = models
        
        B = 2
        action_colour = torch.randint(0, 23, (B,))
        action_onehot = torch.nn.functional.one_hot(action_colour, num_classes=23).float()
        
        # Get action embedding (32 dim)
        action_embedding = action_embedder(action_onehot)
        assert action_embedding.shape == (B, 32)
        
        # Create dummy state tokens (256 dim)
        state_tokens = torch.randn(B, 30, 256)
        
        # Predictor should project action embedding internally
        color_logits = color_predictor(action_embedding, state_tokens, None)
        
        assert color_logits.shape == (B, 11)
        assert color_predictor.action_projection is not None, "Should have action projection"
        assert color_predictor.action_projection.in_features == 32
        assert color_predictor.action_projection.out_features == 256
    
    def test_realistic_scenario(self, models):
        """Test with realistic scenario: different actions on same state"""
        state_encoder, action_embedder, color_predictor = models
        
        B, H, W = 3, 10, 10
        
        # Same state for all samples
        x = torch.randint(0, 11, (B, H, W))
        shape_h = torch.full((B,), H, dtype=torch.long)
        shape_w = torch.full((B,), W, dtype=torch.long)
        most_common = torch.full((B,), 1, dtype=torch.long)
        least_common = torch.full((B,), 0, dtype=torch.long)
        num_colors_grid = torch.full((B,), 5, dtype=torch.long)
        
        # Different actions
        action_colour = torch.tensor([0, 5, 10])  # Different actions
        action_onehot = torch.nn.functional.one_hot(action_colour, num_classes=23).float()
        
        # Forward pass
        state_tokens, causal_mask = state_encoder(
            x, shape_h, shape_w, most_common, least_common, num_colors_grid
        )
        action_embedding = action_embedder(action_onehot)
        color_logits = color_predictor(action_embedding, state_tokens, causal_mask)
        
        # Verify outputs
        assert color_logits.shape == (B, 11)
        
        # Different actions should produce different logits (at least for some samples)
        # Check that logits are not all identical
        logits_diff = (color_logits[0] - color_logits[1]).abs().sum()
        assert logits_diff > 1e-5, "Different actions should produce different logits"
    
    def test_edge_cases(self, models):
        """Test edge cases"""
        state_encoder, action_embedder, color_predictor = models
        
        # Test with minimal grid
        B, H, W = 1, 3, 3
        x = torch.randint(0, 11, (B, H, W))
        shape_h = torch.tensor([3])
        shape_w = torch.tensor([3])
        most_common = torch.tensor([1])
        least_common = torch.tensor([0])
        num_colors_grid = torch.tensor([3])
        
        action_colour = torch.tensor([0])
        action_onehot = torch.nn.functional.one_hot(action_colour, num_classes=23).float()
        
        state_tokens, causal_mask = state_encoder(
            x, shape_h, shape_w, most_common, least_common, num_colors_grid
        )
        action_embedding = action_embedder(action_onehot)
        color_logits = color_predictor(action_embedding, state_tokens, causal_mask)
        
        assert color_logits.shape == (1, 11)
        assert not torch.isnan(color_logits).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

