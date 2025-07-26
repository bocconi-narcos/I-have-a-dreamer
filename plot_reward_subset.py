import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.state_encoder import StateEncoder
from src.models.predictors.reward_predictor import RewardPredictor
from src.data.replay_buffer_dataset import ReplayBufferDataset
from train_reward_predictor import calculate_r2_score, create_and_save_subset_plot

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_trained_models(config, device):
    """Load the trained models from checkpoint."""
    
    # Model parameters
    encoder_params = config['encoder_params']
    latent_dim = config['latent_dim']
    
    # Get image size from encoder params
    image_size = encoder_params.get('image_size', [10, 10])
    input_channels = encoder_params.get('input_channels', 1)
    
    # Initialize models with correct constructor
    state_encoder = StateEncoder(
        image_size=image_size,
        input_channels=input_channels,
        latent_dim=latent_dim,
        encoder_params=encoder_params
    ).to(device)
    
    target_encoder = StateEncoder(
        image_size=image_size,
        input_channels=input_channels,
        latent_dim=latent_dim,
        encoder_params=encoder_params
    ).to(device)
    
    # Initialize reward predictor (using the simple MLP version)
    reward_predictor = RewardPredictor(
        latent_dim=latent_dim,
        hidden_dim=config.get('reward_predictor', {}).get('hidden_dim', 256),
        num_layers=config.get('reward_predictor', {}).get('num_layers', 2),
        dropout=config.get('reward_predictor', {}).get('dropout', 0.1)
    ).to(device)
    
    # Load trained weights
    checkpoint_path = "weights/best_model_reward_predictor.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load state encoders
        if 'state_encoder_state_dict' in checkpoint:
            state_encoder.load_state_dict(checkpoint['state_encoder_state_dict'])
            print("✓ State encoder loaded")
        if 'target_encoder_state_dict' in checkpoint:
            target_encoder.load_state_dict(checkpoint['target_encoder_state_dict'])
            print("✓ Target encoder loaded")
        if 'reward_predictor_state_dict' in checkpoint:
            reward_predictor.load_state_dict(checkpoint['reward_predictor_state_dict'])
            print("✓ Reward predictor loaded")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Available checkpoints:")
        for file in os.listdir("weights/"):
            if "reward" in file:
                print(f"  - {file}")
        return None, None, None
    
    return state_encoder, target_encoder, reward_predictor

def evaluate_model(dataloader, state_encoder, target_encoder, reward_predictor, device):
    """Evaluate model and return predictions and targets."""
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Current state
            state = batch['state'].to(device)
            shape_h = batch.get('shape_h', None)
            shape_w = batch.get('shape_w', None)
            num_colors_grid = batch.get('num_colors_grid', None)
            most_present_color = batch.get('most_present_color', None)
            least_present_color = batch.get('least_present_color', None)

            # Next state
            next_state = batch['next_state'].to(device)
            shape_h_next = batch.get('shape_h_next', None)
            shape_w_next = batch.get('shape_w_next', None)
            num_colors_grid_next = batch.get('num_colors_grid_next', None)
            most_present_color_next = batch.get('most_present_color_next', None)
            least_present_color_next = batch.get('most_present_color_next', None)

            # Target state
            target_state = batch['target_state'].to(device)
            shape_h_target = batch.get('shape_h_target', None)
            shape_w_target = batch.get('shape_w_target', None)
            num_colors_grid_target = batch.get('num_colors_grid_target', None)
            most_present_color_target = batch.get('most_present_color_target', None)
            least_present_color_target = batch.get('least_present_color_target', None)

            # Ground truth reward
            reward = batch['reward'].to(device).float()

            # Add channel dimension if needed
            if state.dim() == 3:
                state = state.unsqueeze(1)
                next_state = next_state.unsqueeze(1)
                target_state = target_state.unsqueeze(1)

            # Encode all three states
            if shape_h is not None:
                latent_t = state_encoder(
                    state.to(torch.long), 
                    shape_h=shape_h.to(device), 
                    shape_w=shape_w.to(device), 
                    num_unique_colors=num_colors_grid.to(device), 
                    most_common_color=most_present_color.to(device), 
                    least_common_color=least_present_color.to(device)
                )
                latent_tp1 = target_encoder(
                    next_state.to(torch.long), 
                    shape_h=shape_h_next.to(device), 
                    shape_w=shape_w_next.to(device), 
                    num_unique_colors=num_colors_grid_next.to(device), 
                    most_common_color=most_present_color_next.to(device), 
                    least_common_color=least_present_color_next.to(device)
                )
                latent_target = target_encoder(
                    target_state.to(torch.long), 
                    shape_h=shape_h_target.to(device), 
                    shape_w=shape_w_target.to(device), 
                    num_unique_colors=num_colors_grid_target.to(device), 
                    most_common_color=most_present_color_target.to(device), 
                    least_common_color=least_present_color_target.to(device)
                )
            else:
                latent_t = state_encoder(state.to(torch.long))
                latent_tp1 = target_encoder(next_state.to(torch.long))
                latent_target = target_encoder(target_state.to(torch.long))

            # Predict reward
            pred_reward = reward_predictor(latent_t, latent_tp1, latent_target)
            
            # Collect predictions and targets
            all_predictions.append(pred_reward.squeeze(-1))
            all_targets.append(reward)
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    return all_predictions, all_targets

def evaluate_and_plot_subset(config, state_encoder, target_encoder, reward_predictor, device, subset_size=3000):
    """Evaluate the model and create subset plots."""
    
    # Buffer setup
    buffer_path = config['buffer_path']
    fast_buffer_path = buffer_path + '.fast.pt'
    
    if not os.path.exists(fast_buffer_path):
        print(f"Fast buffer {fast_buffer_path} not found. Please run preprocessing first.")
        return
    
    # Dataset setup
    dataset = ReplayBufferDataset(
        buffer_path=fast_buffer_path,
        use_ground_truth=False
    )
    
    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Set models to eval mode
    state_encoder.eval()
    target_encoder.eval()
    reward_predictor.eval()
    
    print("Evaluating on validation data...")
    val_predictions, val_targets = evaluate_model(val_loader, state_encoder, target_encoder, reward_predictor, device)
    
    print("Evaluating on training data...")
    train_predictions, train_targets = evaluate_model(train_loader, state_encoder, target_encoder, reward_predictor, device)
    
    # Create and save subset plots
    print("\nCreating subset plots...")
    
    # Validation subset plot
    create_and_save_subset_plot(
        val_targets, val_predictions, 
        subset_size=subset_size, 
        filename="reward_prediction_validation_subset.png"
    )
    
    # Training subset plot
    create_and_save_subset_plot(
        train_targets, train_predictions, 
        subset_size=subset_size, 
        filename="reward_prediction_training_subset.png"
    )
    
    # Calculate overall R² scores
    val_r2 = calculate_r2_score(val_targets, val_predictions)
    train_r2 = calculate_r2_score(train_targets, train_predictions)
    
    print(f"\nOverall R² Scores:")
    print(f"Training: {train_r2:.4f}")
    print(f"Validation: {val_r2:.4f}")
    
    return val_predictions, val_targets, train_predictions, train_targets

def main():
    """Main function to load model and create subset plots."""
    print("Loading configuration...")
    config = load_config()
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load trained models
    print("Loading trained models...")
    state_encoder, target_encoder, reward_predictor = load_trained_models(config, device)
    
    if state_encoder is None:
        print("Failed to load models. Exiting.")
        return
    
    # Evaluate and create subset plots
    print("Starting evaluation and plotting...")
    val_pred, val_targ, train_pred, train_targ = evaluate_and_plot_subset(
        config, state_encoder, target_encoder, reward_predictor, device, subset_size=3000
    )
    
    print("\n✅ Done! Check the generated PNG files:")
    print("  - reward_prediction_validation_subset.png")
    print("  - reward_prediction_training_subset.png")

if __name__ == "__main__":
    main()
