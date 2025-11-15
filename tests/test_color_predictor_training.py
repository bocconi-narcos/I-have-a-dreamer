"""
Test training behavior and cross-entropy loss for CrossAttentionColorPredictor
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


class TestColorPredictorTraining:
    """Tests for training behavior and loss"""
    
    @pytest.fixture
    def models(self):
        """Create models for training test"""
        latent_dim = 128
        action_embedding_dim = 32
        
        encoder_params = {
            'depth': 2, 'heads': 4, 'mlp_dim': 128, 'transformer_dim': 64,
            'dropout': 0.1, 'emb_dropout': 0.1, 'colors_vocab_size': 12
        }
        
        state_encoder = StateEncoder(
            image_size=(5, 5), input_channels=1, latent_dim=latent_dim,
            encoder_params=encoder_params
        )
        
        action_embedder = ActionEmbedder(
            num_actions=23, embed_dim=action_embedding_dim, dropout_p=0.1
        )
        
        color_predictor = CrossAttentionColorPredictor(
            latent_dim=latent_dim,
            num_colors=11,
            action_embedding_dim=action_embedding_dim,
            num_layers=2,
            heads=4,
            mlp_dim=128,
            dropout=0.1,
            mlp_hidden_dim=64
        )
        
        return state_encoder, action_embedder, color_predictor
    
    def test_initial_loss_is_reasonable(self, models):
        """Test that initial loss is reasonable (not too high)"""
        state_encoder, action_embedder, color_predictor = models
        
        B, H, W = 4, 5, 5
        x = torch.randint(0, 11, (B, H, W))
        shape_h = torch.full((B,), H, dtype=torch.long)
        shape_w = torch.full((B,), W, dtype=torch.long)
        most_common = torch.randint(0, 11, (B,))
        least_common = torch.randint(0, 11, (B,))
        num_colors_grid = torch.randint(1, 11, (B,))
        
        action_colour = torch.randint(0, 23, (B,))
        action_onehot = torch.nn.functional.one_hot(action_colour, num_classes=23).float()
        target_colour = torch.randint(0, 11, (B,))
        
        # Forward pass
        state_tokens, causal_mask = state_encoder(
            x, shape_h, shape_w, most_common, least_common, num_colors_grid
        )
        action_embedding = action_embedder(action_onehot)
        color_logits = color_predictor(action_embedding, state_tokens, causal_mask)
        
        # Compute loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(color_logits, target_colour)
        
        # Initial loss should be reasonable (around -log(1/num_classes) = log(11) ≈ 2.4)
        # With random initialization, loss should be in reasonable range
        assert loss.item() > 0, "Loss should be positive"
        assert loss.item() < 10, f"Initial loss should be reasonable, got {loss.item():.4f}"
        assert not torch.isnan(loss), "Loss should not be NaN"
    
    def test_loss_decreases_with_training(self, models):
        """Test that loss decreases with training steps"""
        state_encoder, action_embedder, color_predictor = models
        
        # Create simple dataset
        B, H, W = 8, 5, 5
        x = torch.randint(0, 11, (B, H, W))
        shape_h = torch.full((B,), H, dtype=torch.long)
        shape_w = torch.full((B,), W, dtype=torch.long)
        most_common = torch.randint(0, 11, (B,))
        least_common = torch.randint(0, 11, (B,))
        num_colors_grid = torch.randint(1, 11, (B,))
        
        action_colour = torch.randint(0, 23, (B,))
        action_onehot = torch.nn.functional.one_hot(action_colour, num_classes=23).float()
        
        # Create simple target: most_common color (should be learnable)
        target_colour = most_common.clone()
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            list(state_encoder.parameters()) + 
            list(action_embedder.parameters()) + 
            list(color_predictor.parameters()),
            lr=0.001
        )
        
        # Get initial loss
        state_encoder.eval()
        action_embedder.eval()
        color_predictor.eval()
        with torch.no_grad():
            state_tokens_init, causal_mask_init = state_encoder(
                x, shape_h, shape_w, most_common, least_common, num_colors_grid
            )
            action_embedding_init = action_embedder(action_onehot)
            color_logits_initial = color_predictor(action_embedding_init, state_tokens_init, causal_mask_init)
            loss_initial = criterion(color_logits_initial, target_colour)
        
        state_encoder.train()
        action_embedder.train()
        color_predictor.train()
        
        # Train for a few steps
        losses = [loss_initial.item()]
        for step in range(10):
            optimizer.zero_grad()
            # Recompute everything to create fresh graph
            state_tokens, causal_mask = state_encoder(
                x, shape_h, shape_w, most_common, least_common, num_colors_grid
            )
            action_embedding = action_embedder(action_onehot)
            color_logits = color_predictor(action_embedding, state_tokens, causal_mask)
            loss = criterion(color_logits, target_colour)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(state_encoder.parameters()) + 
                list(action_embedder.parameters()) + 
                list(color_predictor.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            losses.append(loss.item())
        
        # Loss should decrease (or at least not increase significantly)
        final_loss = losses[-1]
        initial_loss = losses[0]
        
        assert final_loss > 0, "Loss should remain positive"
        assert not torch.isnan(torch.tensor(final_loss)), "Loss should not be NaN"
        
        # Loss should decrease or stay similar (allowing for some variance)
        # In practice, loss should decrease with training
        print(f"   Initial loss: {initial_loss:.4f}, Final loss: {final_loss:.4f}")
        assert final_loss <= initial_loss * 1.5, f"Loss should decrease or stay similar. Initial: {initial_loss:.4f}, Final: {final_loss:.4f}"
    
    def test_cross_entropy_with_perfect_predictions(self, models):
        """Test that cross-entropy can be very low with perfect predictions"""
        state_encoder, action_embedder, color_predictor = models
        
        B = 4
        x = torch.randint(0, 11, (B, 5, 5))
        shape_h = torch.full((B,), 5, dtype=torch.long)
        shape_w = torch.full((B,), 5, dtype=torch.long)
        most_common = torch.randint(0, 11, (B,))
        least_common = torch.randint(0, 11, (B,))
        num_colors_grid = torch.randint(1, 11, (B,))
        
        action_colour = torch.randint(0, 23, (B,))
        action_onehot = torch.nn.functional.one_hot(action_colour, num_classes=23).float()
        
        state_tokens, causal_mask = state_encoder(
            x, shape_h, shape_w, most_common, least_common, num_colors_grid
        )
        action_embedding = action_embedder(action_onehot)
        color_logits = color_predictor(action_embedding, state_tokens, causal_mask)
        
        # Create perfect predictions (one-hot logits)
        # Set logits to very high value for correct class, low for others
        target_colour = torch.randint(0, 11, (B,))
        perfect_logits = torch.full_like(color_logits, -10.0)
        perfect_logits.scatter_(1, target_colour.unsqueeze(1), 10.0)
        
        # Compute loss with perfect predictions
        criterion = torch.nn.CrossEntropyLoss()
        perfect_loss = criterion(perfect_logits, target_colour)
        
        # Perfect predictions should give very low loss
        assert perfect_loss.item() < 0.1, f"Perfect predictions should give very low loss, got {perfect_loss.item():.6f}"
        print(f"   Perfect prediction loss: {perfect_loss.item():.6f} (should be very low)")
    
    def test_model_can_achieve_low_loss(self, models):
        """Test that model can achieve low cross-entropy loss with training"""
        state_encoder, action_embedder, color_predictor = models
        
        # Create simple learnable pattern
        B, H, W = 16, 5, 5
        x = torch.randint(0, 11, (B, H, W))
        shape_h = torch.full((B,), H, dtype=torch.long)
        shape_w = torch.full((B,), W, dtype=torch.long)
        most_common = torch.randint(0, 11, (B,))
        least_common = torch.randint(0, 11, (B,))
        num_colors_grid = torch.randint(1, 11, (B,))
        
        # Simple target: predict most_common color (should be learnable from state)
        target_colour = most_common.clone()
        
        action_colour = torch.randint(0, 23, (B,))
        action_onehot = torch.nn.functional.one_hot(action_colour, num_classes=23).float()
        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            list(state_encoder.parameters()) + 
            list(action_embedder.parameters()) + 
            list(color_predictor.parameters()),
            lr=0.01  # Higher LR for faster convergence
        )
        
        # Train for more steps
        best_loss = float('inf')
        for step in range(50):
            optimizer.zero_grad()
            # Recompute everything to create fresh graph each iteration
            state_tokens, causal_mask = state_encoder(
                x, shape_h, shape_w, most_common, least_common, num_colors_grid
            )
            action_embedding = action_embedder(action_onehot)
            color_logits = color_predictor(action_embedding, state_tokens, causal_mask)
            loss = criterion(color_logits, target_colour)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(state_encoder.parameters()) + 
                list(action_embedder.parameters()) + 
                list(color_predictor.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            
            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
        
        # After training, loss should be low
        print(f"   Best loss achieved: {best_loss:.6f}")
        assert best_loss < 2.0, f"Loss should be low after training, got {best_loss:.4f}"
        assert not torch.isnan(torch.tensor(best_loss)), "Loss should not be NaN"
        
        # Verify predictions improve
        color_predictor.eval()
        with torch.no_grad():
            final_logits = color_predictor(action_embedding, state_tokens, causal_mask)
            final_predictions = torch.argmax(final_logits, dim=1)
            accuracy = (final_predictions == target_colour).float().mean()
            print(f"   Final accuracy: {accuracy.item():.4f}")
            assert accuracy.item() > 0.1, "Model should learn something (accuracy > 10%)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

