"""
Comprehensive tests for CrossAttentionColorPredictor
"""

import torch
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.predictors.color_predictor import CrossAttentionColorPredictor, PreNormCrossAttentionBlock
from models.state_encoder import StateEncoder
from models.action_embed import ActionEmbedder


class TestCrossAttentionColorPredictor:
    """Tests for CrossAttentionColorPredictor"""
    
    @pytest.fixture
    def predictor(self):
        """Create a CrossAttentionColorPredictor instance"""
        return CrossAttentionColorPredictor(
            latent_dim=128,
            num_colors=11,
            action_embedding_dim=32,
            num_layers=2,
            heads=4,
            mlp_dim=128,
            dropout=0.1,
            mlp_hidden_dim=64
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample input data"""
        B, H, W = 2, 5, 5
        num_tokens = 5 + H * W  # 30 tokens
        
        # State tokens
        state_tokens = torch.randn(B, num_tokens, 128)
        
        # Action embedding
        action_embedding = torch.randn(B, 32)
        
        # Causal mask (all tokens valid)
        causal_mask = torch.zeros(B, num_tokens, num_tokens, dtype=torch.bool)
        
        return state_tokens, action_embedding, causal_mask
    
    def test_forward_pass(self, predictor, sample_data):
        """Test basic forward pass"""
        state_tokens, action_embedding, causal_mask = sample_data
        
        logits = predictor(action_embedding, state_tokens, causal_mask)
        
        assert logits.shape == (2, 11), f"Expected (2, 11), got {logits.shape}"
        assert logits.dtype == torch.float32
    
    def test_forward_without_mask(self, predictor, sample_data):
        """Test forward pass without causal mask"""
        state_tokens, action_embedding, _ = sample_data
        
        logits = predictor(action_embedding, state_tokens, None)
        
        assert logits.shape == (2, 11)
    
    def test_action_projection(self):
        """Test that action embedding is projected correctly"""
        predictor = CrossAttentionColorPredictor(
            latent_dim=128,
            num_colors=11,
            action_embedding_dim=32,  # Different from latent_dim
            num_layers=1,
            heads=2,
            mlp_dim=64,
            dropout=0.0
        )
        
        assert predictor.action_projection is not None, "Should have action projection"
        assert predictor.action_projection.in_features == 32
        assert predictor.action_projection.out_features == 128
        
        # Test forward with different action dim
        B = 2
        action_embedding = torch.randn(B, 32)
        state_tokens = torch.randn(B, 10, 128)
        
        logits = predictor(action_embedding, state_tokens, None)
        assert logits.shape == (B, 11)
    
    def test_no_action_projection_when_same_dim(self):
        """Test that no projection is needed when dimensions match"""
        predictor = CrossAttentionColorPredictor(
            latent_dim=128,
            num_colors=11,
            action_embedding_dim=128,  # Same as latent_dim
            num_layers=1,
            heads=2,
            mlp_dim=64,
            dropout=0.0
        )
        
        assert predictor.action_projection is None, "Should not have action projection"
    
    def test_causal_mask_conversion(self, predictor, sample_data):
        """Test causal mask to padding mask conversion"""
        state_tokens, action_embedding, causal_mask = sample_data
        
        # Test 2D mask conversion
        padding_mask = predictor._convert_causal_mask_to_padding_mask(
            causal_mask, state_tokens.shape[1]
        )
        
        assert padding_mask.shape == (2, 30), f"Expected (2, 30), got {padding_mask.shape}"
        assert padding_mask.dtype == torch.bool
        
        # Test 1D mask (already padding mask)
        padding_mask_1d = torch.zeros(2, 30, dtype=torch.bool)
        result = predictor._convert_causal_mask_to_padding_mask(
            padding_mask_1d, 30
        )
        assert result.shape == (2, 30)
    
    def test_gradient_flow(self, predictor, sample_data):
        """Test that gradients flow through the model"""
        state_tokens, action_embedding, causal_mask = sample_data
        
        logits = predictor(action_embedding, state_tokens, causal_mask)
        loss = logits.sum()
        loss.backward()
        
        has_gradients = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in predictor.parameters()
            if p.requires_grad
        )
        
        assert has_gradients, "Gradients should flow through model"
    
    def test_different_batch_sizes(self, predictor):
        """Test with different batch sizes"""
        for B in [1, 2, 4]:
            state_tokens = torch.randn(B, 30, 128)
            action_embedding = torch.randn(B, 32)
            
            logits = predictor(action_embedding, state_tokens, None)
            assert logits.shape == (B, 11)
    
    def test_different_num_tokens(self, predictor):
        """Test with different number of tokens"""
        B = 2
        action_embedding = torch.randn(B, 32)
        
        for num_tokens in [10, 20, 30]:
            state_tokens = torch.randn(B, num_tokens, 128)
            causal_mask = torch.zeros(B, num_tokens, num_tokens, dtype=torch.bool)
            
            logits = predictor(action_embedding, state_tokens, causal_mask)
            assert logits.shape == (B, 11)


class TestPreNormCrossAttentionBlock:
    """Tests for PreNormCrossAttentionBlock"""
    
    @pytest.fixture
    def block(self):
        """Create a PreNormCrossAttentionBlock"""
        return PreNormCrossAttentionBlock(
            latent_dim=128,
            heads=4,
            mlp_dim=256,
            dropout=0.1
        )
    
    def test_forward_pass(self, block):
        """Test forward pass through block"""
        B = 2
        query = torch.randn(B, 1, 128)  # Action as query
        key_value = torch.randn(B, 10, 128)  # State tokens as key/value
        
        output = block(query, key_value)
        
        assert output.shape == (B, 1, 128)
    
    def test_with_padding_mask(self, block):
        """Test with padding mask"""
        B = 2
        query = torch.randn(B, 1, 128)
        key_value = torch.randn(B, 10, 128)
        padding_mask = torch.zeros(B, 10, dtype=torch.bool)  # All valid
        
        output = block(query, key_value, key_padding_mask=padding_mask)
        
        assert output.shape == (B, 1, 128)
    
    def test_gradient_flow(self, block):
        """Test gradient flow"""
        B = 2
        query = torch.randn(B, 1, 128)
        key_value = torch.randn(B, 10, 128)
        
        output = block(query, key_value)
        loss = output.sum()
        loss.backward()
        
        has_gradients = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in block.parameters()
            if p.requires_grad
        )
        
        assert has_gradients, "Gradients should flow"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

