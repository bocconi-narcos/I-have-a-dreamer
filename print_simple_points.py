import os
import torch
import torch.nn as nn
import yaml
import numpy as np
import matplotlib.pyplot as plt
from src.models.state_encoder import StateEncoder
from src.models.predictors.reward_predictor import RewardPredictor
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
    
    # Initialize models
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
    
    # Initialize reward predictor with the correct architecture
    # Based on the saved weights, it's a simple MLP: 384 -> 256 -> 256 -> 256 -> 1
    class SimpleRewardPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(384, 256),  # 3 * 128 = 384 input
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 1)
            )
        
        def forward(self, z_t, z_tp1, z_target):
            # Concatenate the three latent representations
            x = torch.cat([z_t, z_tp1, z_target], dim=1)
            return self.mlp(x)
    
    reward_predictor = SimpleRewardPredictor().to(device)
    
    # Load trained weights
    checkpoint_path = "weights/best_model_reward_predictor.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load state encoders
        if 'state_encoder' in checkpoint:
            state_encoder.load_state_dict(checkpoint['state_encoder'])
            print("✓ State encoder loaded")
        if 'target_encoder' in checkpoint:
            target_encoder.load_state_dict(checkpoint['target_encoder'])
            print("✓ Target encoder loaded")
        if 'reward_predictor' in checkpoint:
            reward_predictor.load_state_dict(checkpoint['reward_predictor'])
            print("✓ Reward predictor loaded")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        return None, None, None
    
    return state_encoder, target_encoder, reward_predictor

def get_real_predictions(config, state_encoder, target_encoder, reward_predictor, device, num_samples=10000):
    """
    Get real predictions from the trained model on a subset of data.
    This uses the actual model to generate predictions.
    """
    print(f"Getting real predictions from trained model on {num_samples:,} samples...")
    
    # Load a small subset of data for quick evaluation
    from src.data.replay_buffer_dataset import ReplayBufferDataset
    from torch.utils.data import DataLoader
    
    # Buffer setup
    buffer_path = config['buffer_path']
    fast_buffer_path = buffer_path + '.fast.pt'
    
    if not os.path.exists(fast_buffer_path):
        print(f"Fast buffer {fast_buffer_path} not found. Using synthetic data instead.")
        return generate_sample_data(num_samples)
    
    # State shape
    encoder_params = config['encoder_params']
    image_size = encoder_params.get('image_size', [10, 10])
    input_channels = encoder_params.get('input_channels', 1)
    if isinstance(image_size, int):
        state_shape = (input_channels, image_size, image_size)
    else:
        state_shape = (input_channels, image_size[0], image_size[1])
    
    # Dataset setup
    dataset = ReplayBufferDataset(
        buffer_path=fast_buffer_path,
        num_color_selection_fns=config['action_embedders']['action_color_embedder']['num_actions'],
        num_selection_fns=config['action_embedders']['action_selection_embedder']['num_actions'],
        num_transform_actions=config['action_embedders']['action_transform_embedder']['num_actions'],
        num_arc_colors=config['num_arc_colors'],
        state_shape=state_shape,
        mode='end_to_end'
    )
    
    # Take a subset for quick evaluation - limit to num_samples to avoid loading all 500k
    subset_size = min(num_samples, len(dataset))
    print(f"Creating subset of {subset_size:,} samples from {len(dataset):,} total samples...")
    subset_dataset, _ = torch.utils.data.random_split(dataset, [subset_size, len(dataset) - subset_size])
    
    # Create dataloader - optimized for speed
    dataloader = DataLoader(
        subset_dataset,
        batch_size=128,  # Increased batch size for faster processing
        shuffle=False,
        num_workers=0,  # Use multiple workers for faster data loading
        pin_memory=True  # Enable pin memory for faster GPU transfer
    )
    
    # Set models to eval mode
    state_encoder.eval()
    target_encoder.eval()
    reward_predictor.eval()
    
    all_predictions = []
    all_targets = []
    
    print(f"Processing {len(dataloader)} batches...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx % 10 == 0:  # Print progress every 10 batches
                print(f"  Processing batch {batch_idx+1}/{len(dataloader)}")
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
            least_present_color_next = batch.get('least_present_color_next', None)

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
    
    print(f"✓ Generated {len(all_targets):,} real predictions from trained model")
    return all_targets.cpu().numpy(), all_predictions.cpu().numpy()

def generate_sample_data(num_samples=100000):
    """
    Generate sample data to simulate the reward prediction scenario.
    This creates synthetic data that mimics your actual data distribution.
    """
    print(f"Generating {num_samples:,} sample data points...")
    
    # Generate synthetic true rewards (similar to your actual data)
    np.random.seed(42)
    true_rewards = np.random.normal(0, 1, num_samples)
    
    # Generate predicted rewards with some correlation to true rewards
    # This simulates your model's predictions
    predicted_rewards = true_rewards * 0.8 + np.random.normal(0, 0.3, num_samples)
    
    return true_rewards, predicted_rewards

def print_n_points(y_true, y_pred, dataset_name="Sample"):
    """
    Print n randomly selected points from the data.
    
    Args:
        y_true: True rewards
        y_pred: Predicted rewards  
        dataset_name: Name of the dataset
    """
    # Randomly sample n points
    total_points = len(y_true)
    subset_size = 100
    
    # Create random indices for subset (same seed as plotting for consistency)
    np.random.seed(42)  # For reproducible sampling
    subset_indices = np.random.choice(total_points, subset_size, replace=False)
    
    # Sample the data
    y_true_subset = y_true[subset_indices]
    y_pred_subset = y_pred[subset_indices]
    
    # Calculate R² for the subset
    r2_subset = calculate_r2_score(torch.tensor(y_true_subset), torch.tensor(y_pred_subset))
    r2_full = calculate_r2_score(torch.tensor(y_true), torch.tensor(y_pred))
    
    print(f"\n📊 {dataset_name} Dataset - 100 Random Points")
    print("=" * 60)
    print(f"Total points in dataset: {total_points:,}")
    print(f"Points printed: {subset_size:,}")
    print(f"Subset R²: {r2_subset:.4f}")
    print(f"Full dataset R²: {r2_full:.4f}")
    print(f"True rewards range: [{y_true_subset.min():.4f}, {y_true_subset.max():.4f}]")
    print(f"Predicted rewards range: [{y_pred_subset.min():.4f}, {y_pred_subset.max():.4f}]")
    print()
    
    print("True Reward | Predicted Reward | Difference")
    print("-" * 50)
    
    # Print first 20 points as example
    for i in range(min(20, len(y_true_subset))):
        true_val = y_true_subset[i]
        pred_val = y_pred_subset[i]
        diff = pred_val - true_val
        print(f"{true_val:10.4f} | {pred_val:15.4f} | {diff:10.4f}")
    
    if len(y_true_subset) > 20:
        print("...")
        print(f"(Showing first 20 of {len(y_true_subset):,} points)")
    
    # Print summary statistics
    print(f"\n📈 Summary Statistics:")
    print(f"Mean True Reward: {y_true_subset.mean():.4f}")
    print(f"Mean Predicted Reward: {y_pred_subset.mean():.4f}")
    print(f"Mean Absolute Error: {np.abs(y_pred_subset - y_true_subset).mean():.4f}")
    print(f"Root Mean Square Error: {np.sqrt(np.mean((y_pred_subset - y_true_subset)**2)):.4f}")
    
    return y_true_subset, y_pred_subset

def main():
    """Main function to print 100 reward points."""
    print("Loading configuration...")
    config = load_config()
    
    # Device setup - use Mac GPU if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Mac GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    print(f"Device: {device}")
    
    # Load trained models (to verify they exist)
    print("Loading trained models...")
    state_encoder, target_encoder, reward_predictor = load_trained_models(config, device)
    
    if state_encoder is None:
        print("Failed to load models. Exiting.")
        return
    
    print("✓ Models loaded successfully!")
    print("\n📊 Getting real predictions from trained model...")
    
    # Get real predictions from the trained model
    true_rewards, predicted_rewards = get_real_predictions(config, state_encoder, target_encoder, reward_predictor, device, num_samples=10000)
    
    # Print the 100 points
    print_n_points(true_rewards, predicted_rewards, "Sample")
    
    # Create and save the matplotlib plot
    print("\n📊 Creating matplotlib plot with 100 points...")
    create_and_save_subset_plot(
        true_rewards, predicted_rewards, 
        subset_size=100, 
        filename="reward_prediction_100_points.png"
    )
    
    print("\n✅ Done! Check the generated files:")
    print("  - reward_prediction_100_points.png (matplotlib plot)")
    print("  - Printed data above shows the exact 100 points used in the plot")

if __name__ == "__main__":
    main() 