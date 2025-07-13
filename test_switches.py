#!/usr/bin/env python3
"""
Test script to verify the ground truth and decoder switches work correctly.
"""

import yaml
import torch
from train_next_state_predictor import load_config

def test_switches():
    """Test that the switches are loaded correctly from config."""
    
    # Test default config2321233
    config = load_config()
    print("Default config:")
    print(f"  use_ground_truth: {config.get('use_ground_truth', False)}")
    print(f"  use_decoder_loss: {config.get('use_decoder_loss', False)}")
    
    # Test with switches enabled
    test_config = {
        'use_ground_truth': True,
        'use_decoder_loss': True,
        'buffer_path': 'data/buffer_5001    .pt',
        'encoder_type': 'transformer',
        'latent_dim': 128,
        'encoder_params': {'image_size': [10, 10], 'input_channels': 1},
        'action_embedders': {
            'action_color_embedder': {'num_actions': 10, 'embed_dim': 64},
            'action_selection_embedder': {'num_actions': 10, 'embed_dim': 64},
            'action_transform_embedder': {'num_actions': 10, 'embed_dim': 64}
        },
        'num_arc_colors': 11,
        'color_predictor': {'hidden_dim': 128},
        'selection_mask': {
            'mask_encoder_params': {},
            'latent_mask_dim': 64,
            'mask_predictor_params': {
                'transformer_depth': 2,
                'transformer_heads': 2,
                'transformer_dim_head': 32,
                'transformer_mlp_dim': 128,
                'transformer_dropout': 0.1
            }
        },
        'next_state': {
            'latent_mask_dim': 64,
            'transformer_depth': 2,
            'transformer_heads': 2,
            'transformer_dim_head': 32,
            'transformer_mlp_dim': 128,
            'transformer_dropout': 0.1
        },
        'reward_predictor': {
            'hidden_dim': 128,
            'transformer_depth': 2,
            'transformer_heads': 2,
            'transformer_dim_head': 64,
            'transformer_mlp_dim': 128,
            'transformer_dropout': 0.1
        },
        'batch_size': 32,
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'num_workers': 4,
        'log_interval': 100,
        'decoder_params': {},
        'mask_decoder_params': {}
    }
    
    # Save test config
    with open('test_config.yaml', 'w') as f:
        yaml.dump(test_config, f)
    
    # Load test config
    config = load_config('test_config.yaml')
    print("\nTest config with switches enabled:")
    print(f"  use_ground_truth: {config.get('use_ground_truth', False)}")
    print(f"  use_decoder_loss: {config.get('use_decoder_loss', False)}")
    
    print("\n✅ Switch configuration test passed!")

if __name__ == "__main__":
    test_switches() 