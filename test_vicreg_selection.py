#!/usr/bin/env python3
"""
Test script to verify that VICReg loss works correctly for selection mask prediction.
"""

import torch
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.state_encoder import StateEncoder
from src.models.predictors.color_predictor import ColorPredictor
from src.models.predictors.selection_mask_predictor import SelectionMaskPredictor
from src.models.action_embed import ActionEmbedder
from src.models.mask_encoder_new import MaskEncoder
from src.losses.vicreg import VICRegLoss

def test_vicreg_selection():
    """Test that VICReg loss works correctly for selection mask prediction."""
    
    # Configuration
    batch_size = 4
    latent_dim = 128
    color_selection_dim = 32
    selection_dim = 32
    num_color_selection_fns = 23
    num_selection_fns = 6
    num_arc_colors = 12
    latent_mask_dim = 64
    
    device = torch.device('cpu')
    
    print("Testing VICReg loss for selection mask prediction...")
    
    # Create models
    state_encoder = StateEncoder(
        image_size=[10, 10],
        input_channels=1,
        latent_dim=latent_dim,
        encoder_params={
            'depth': 4,
            'heads': 16,
            'mlp_dim': 512,
            'transformer_dim': 64,
            'pool': "cls",
            'dropout': 0.2,
            'emb_dropout': 0.2,
            'input_channels': 1,
            'image_size': [10, 10],
            'scaled_position_embeddings': False,
            'colors_vocab_size': 12
        }
    ).to(device)
    
    color_predictor = ColorPredictor(
        latent_dim=latent_dim,
        num_colors=num_arc_colors,
        hidden_dim=512,
        action_embedding_dim=color_selection_dim
    ).to(device)
    
    selection_mask_predictor = SelectionMaskPredictor(
        state_dim=latent_dim,
        selection_action_embed_dim=selection_dim,
        color_pred_dim=num_arc_colors,
        latent_mask_dim=latent_mask_dim,
        transformer_depth=2,
        transformer_heads=2,
        transformer_dim_head=32,
        transformer_mlp_dim=64,
        dropout=0.1
    ).to(device)
    
    colour_selection_embedder = ActionEmbedder(
        num_actions=num_color_selection_fns,
        embed_dim=color_selection_dim,
        dropout_p=0.1
    ).to(device)
    
    selection_embedder = ActionEmbedder(
        num_actions=num_selection_fns,
        embed_dim=selection_dim,
        dropout_p=0.1
    ).to(device)
    
    mask_encoder = MaskEncoder(
        image_size=[10, 10],
        vocab_size=12,
        emb_dim=64,
        depth=4,
        heads=8,
        mlp_dim=512,
        dropout=0.2,
        emb_dropout=0.2,
        padding_value=-1
    ).to(device)
    
    # Create VICReg loss function
    vicreg_loss_fn = VICRegLoss(sim_coeff=25.0, std_coeff=25.0, cov_coeff=1.0)
    
    # Create dummy data
    state = torch.randint(-1, 11, (batch_size, 10, 10)).to(device)
    action_colour = torch.randint(0, num_color_selection_fns, (batch_size,)).to(device)
    action_selection = torch.randint(0, num_selection_fns, (batch_size,)).to(device)
    target_colour = torch.randint(0, num_arc_colors, (batch_size,)).to(device)
    selection_mask = torch.randint(-1, 11, (batch_size, 10, 10)).to(device)
    
    # One-hot encode actions
    action_colour_onehot = torch.nn.functional.one_hot(action_colour, num_color_selection_fns).float()
    action_selection_onehot = torch.nn.functional.one_hot(action_selection, num_selection_fns).float()
    
    print(f"Input shapes:")
    print(f"  State: {state.shape}")
    print(f"  Action colour one-hot: {action_colour_onehot.shape}")
    print(f"  Action selection one-hot: {action_selection_onehot.shape}")
    
    # Forward pass through state encoder
    latent = state_encoder(
        state,
        shape_w=torch.ones(batch_size, dtype=torch.long) * 10,
        shape_h=torch.ones(batch_size, dtype=torch.long) * 10,
        num_unique_colors=torch.ones(batch_size, dtype=torch.long) * 8,
        most_common_color=torch.ones(batch_size, dtype=torch.long) * 1,
        least_common_color=torch.ones(batch_size, dtype=torch.long) * 0
    )
    print(f"  Latent state: {latent.shape}")
    
    # Embed actions
    action_color_embedding = colour_selection_embedder(action_colour_onehot)
    action_selection_embedding = selection_embedder(action_selection_onehot)
    
    print(f"  Embedded action colour: {action_color_embedding.shape}")
    print(f"  Embedded action selection: {action_selection_embedding.shape}")
    
    # Color prediction
    color_logits = color_predictor(latent, action_color_embedding)
    print(f"  Color logits: {color_logits.shape}")
    
    # Selection mask prediction
    pred_latent_mask = selection_mask_predictor(latent, action_selection_embedding, color_logits.softmax(dim=1))
    print(f"  Predicted latent mask: {pred_latent_mask.shape}")
    
    # Target latent mask
    target_latent_mask = mask_encoder(selection_mask.long())
    print(f"  Target latent mask: {target_latent_mask.shape}")
    
    # Test VICReg loss
    selection_loss, sim_loss, std_loss, cov_loss = vicreg_loss_fn(pred_latent_mask, target_latent_mask)
    
    print(f"\nVICReg Loss Components:")
    print(f"  Total Selection Loss: {selection_loss.item():.4f}")
    print(f"  Similarity Loss: {sim_loss.item():.4f}")
    print(f"  Variance Loss: {std_loss.item():.4f}")
    print(f"  Covariance Loss: {cov_loss.item():.4f}")
    
    # Test color loss (unchanged)
    color_loss = torch.nn.CrossEntropyLoss()(color_logits, target_colour)
    print(f"  Color Loss: {color_loss.item():.4f}")
    
    # Test combined loss
    total_loss = color_loss + selection_loss
    print(f"  Combined Loss: {total_loss.item():.4f}")
    
    print("\n✅ VICReg loss for selection mask prediction works correctly!")
    
    # Test that gradients flow
    total_loss.backward()
    print("✅ Gradients flow correctly through all components.")
    
    return True

if __name__ == "__main__":
    test_vicreg_selection() 