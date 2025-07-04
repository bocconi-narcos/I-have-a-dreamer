#!/usr/bin/env python3
"""
Train Step Distance Encoder

This script trains a state encoder to learn representations where the cosine similarity
between state encodings correlates with the step distance between states. The training
objective uses an exponential decay function: similarity = exp(-alpha * step_distance).

Key features:
- Uses the same state encoder architecture as the autoencoder
- Can optionally load pre-trained autoencoder weights as initialization
- Filters out invalid step distances (-1 values) from the training data
- Includes comprehensive logging and validation with W&B
- Implements early stopping based on validation loss

Usage:
    python train_step_distance_encoder.py

The script will automatically load configuration from config_autoencoder.yaml and
use the replay buffer specified in the config.
"""

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
import wandb
from src.models.state_encoder import StateEncoder
from src.data.replay_buffer_dataset import ReplayBufferDataset
import torch.nn.functional as F
import h5py

def load_config(config_path="config_autoencoder.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class StepDistanceDataset(ReplayBufferDataset):
    """Extended dataset that includes target_state and step_distance_to_target"""
    
    def _load_hdf5_buffer(self, buffer_path):
        """Load replay buffer data from an HDF5 file including target state and step distance."""
        buffer = []
        with h5py.File(buffer_path, 'r') as f:
            num_transitions = f['state'].shape[0]
            
            for i in range(num_transitions):
                # Skip transitions with invalid step distance (-1)
                step_distance = f['step_distance_to_target'][i]
                if step_distance < 0:
                    continue
                    
                transition = {
                    'state': f['state'][i],
                    'action': {
                        'colour': f['action_colour'][i],
                        'selection': f['action_selection'][i],
                        'transform': f['action_transform'][i]
                    },
                    'selection_mask': f['selection_mask'][i],
                    'next_state': f['next_state'][i],
                    'target_state': f['target_state'][i],
                    'step_distance_to_target': step_distance,
                    'colour': f['colour'][i],
                    'reward': f['reward'][i],
                    'done': f['done'][i],
                    'transition_type': f['transition_type'][i],
                    'shape_h': f['shape_h'][i],
                    'shape_w': f['shape_w'][i],
                    'num_colors_grid': f['num_colors_grid'][i],
                    'most_present_color': f['most_present_color'][i],
                    'least_present_color': f['least_present_color'][i]
                }
                buffer.append(transition)
        return buffer
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset including target state and step distance."""
        transition = self.buffer[idx]
        
        # Extract states and convert to tensor
        state = torch.tensor(transition['state'], dtype=torch.long)
        target_state = torch.tensor(transition['target_state'], dtype=torch.long)
        step_distance = torch.tensor(transition['step_distance_to_target'], dtype=torch.float32)
        
        # Extract grid statistics for state
        shape_h = torch.tensor(transition['shape_h'], dtype=torch.long)
        shape_w = torch.tensor(transition['shape_w'], dtype=torch.long)
        num_colors_grid = torch.tensor(transition['num_colors_grid'], dtype=torch.long)
        most_present_color = torch.tensor(transition['most_present_color'], dtype=torch.long)
        least_present_color = torch.tensor(transition['least_present_color'], dtype=torch.long)
        
        # For target state, we'll use the same statistics as the current state
        # This is a simplification - ideally we'd compute target state statistics separately
        sample = {
            'state': state,
            'target_state': target_state,
            'step_distance_to_target': step_distance,
            'shape_h': shape_h,
            'shape_w': shape_w,
            'num_colors_grid': num_colors_grid,
            'most_present_color': most_present_color,
            'least_present_color': least_present_color,
        }
        
        return sample

def step_distance_loss(state_encoding, target_encoding, step_distance, alpha=1.0):
    """
    Compute step distance loss based on cosine similarity and expected similarity.
    
    This loss function trains the encoder to produce representations where:
    - States that are 0 steps apart (identical) have cosine similarity = 1
    - States that are many steps apart have cosine similarity approaching 0
    - The decay follows an exponential: exp(-alpha * step_distance)
    
    Args:
        state_encoding: Encoded state representation [B, latent_dim]
        target_encoding: Encoded target state representation [B, latent_dim]
        step_distance: Step distance to target [B] (non-negative values)
        alpha: Decay parameter controlling how quickly similarity decreases with distance
    
    Returns:
        loss: MSE loss between cosine similarity and expected similarity
        metrics: Dictionary with loss components for monitoring
    """
    # Normalize encodings for cosine similarity
    state_norm = F.normalize(state_encoding, p=2, dim=-1)
    target_norm = F.normalize(target_encoding, p=2, dim=-1)
    
    # Compute cosine similarity
    cosine_sim = (state_norm * target_norm).sum(dim=-1)  # [B]
    
    # Expected similarity: exp(-alpha * step_distance)
    # When step_distance = 0, exp(-0) = 1 (perfect similarity)
    # When step_distance -> inf, exp(-inf) -> 0 (no similarity)
    expected_sim = torch.exp(-alpha * step_distance)
    
    # MSE loss between actual and expected similarity
    loss = F.mse_loss(cosine_sim, expected_sim)
    
    metrics = {
        'cosine_similarity': cosine_sim.mean().item(),
        'expected_similarity': expected_sim.mean().item(),
        'step_distance': step_distance.mean().item()
    }
    
    return loss, metrics

def evaluate(encoder, dataloader, device, alpha=1.0):
    encoder.eval()
    total_loss = 0
    total_cosine_sim = 0
    total_expected_sim = 0
    total_step_distance = 0
    total = 0
    
    with torch.no_grad():
        for batch in dataloader:
            state = batch['state'].to(device)
            target_state = batch['target_state'].to(device)
            step_distance = batch['step_distance_to_target'].to(device)
            shape_w = batch['shape_w'].to(device)
            shape_h = batch['shape_h'].to(device)
            num_colors_grid = batch['num_colors_grid'].to(device)
            most_present_color = batch['most_present_color'].to(device)
            least_present_color = batch['least_present_color'].to(device)

            # Encode both states
            state_encoding = encoder(
                state,
                shape_w=shape_w,
                shape_h=shape_h,
                num_unique_colors=num_colors_grid,
                most_common_color=most_present_color,
                least_common_color=least_present_color
            )
            
            target_encoding = encoder(
                target_state,
                shape_w=shape_w,
                shape_h=shape_h,
                num_unique_colors=num_colors_grid,
                most_common_color=most_present_color,
                least_common_color=least_present_color
            )
            
            loss, metrics = step_distance_loss(state_encoding, target_encoding, step_distance, alpha)
            
            batch_size = state.size(0)
            total_loss += loss.item() * batch_size
            total_cosine_sim += metrics['cosine_similarity'] * batch_size
            total_expected_sim += metrics['expected_similarity'] * batch_size
            total_step_distance += metrics['step_distance'] * batch_size
            total += batch_size

    avg_loss = total_loss / total
    avg_cosine_sim = total_cosine_sim / total
    avg_expected_sim = total_expected_sim / total
    avg_step_distance = total_step_distance / total

    return avg_loss, {
        'cosine_similarity': avg_cosine_sim,
        'expected_similarity': avg_expected_sim,
        'step_distance': avg_step_distance
    }

def train_step_distance_encoder():
    config = load_config()
    buffer_path = config['buffer_path']
    latent_dim = config['latent_dim']
    encoder_params = config['encoder_params']
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    num_workers = config['num_workers']
    log_interval = config['log_interval']
    
    # Step distance specific parameters
    alpha = 1.0  # Decay parameter for expected similarity

    image_size = encoder_params.get('image_size', [10, 10])
    input_channels = encoder_params.get('input_channels', 1)
    if isinstance(image_size, int):
        state_shape = (input_channels, image_size, image_size)
    else:
        state_shape = (input_channels, image_size[0], image_size[1])

    # Use the extended dataset
    dataset = StepDistanceDataset(
        buffer_path=buffer_path,
        num_color_selection_fns=config['num_color_selection_fns'],
        num_selection_fns=config['num_selection_fns'],
        num_transform_actions=config['num_transform_actions'],
        num_arc_colors=11,
        state_shape=state_shape,
        mode='full'
    )

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Initialize encoder
    state_encoder = StateEncoder(
        image_size=image_size,
        input_channels=input_channels,
        latent_dim=latent_dim,
        encoder_params=encoder_params
    ).to(device)

    # Load pre-trained encoder weights if available
    #pretrained_path = 'best_model_autoencoder.pth'
    #if os.path.exists(pretrained_path):
    #    print(f"Loading pre-trained encoder from {pretrained_path}")
    #    checkpoint = torch.load(pretrained_path, map_location=device)
    #    state_encoder.load_state_dict(checkpoint['state_encoder'])
    #else:
    #    print("No pre-trained encoder found, training from scratch")

    optimizer = optim.AdamW(state_encoder.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 50
    save_path = 'best_model_step_distance_encoder.pth'
    
    # Initialize wandb
    wandb_config = config.copy()
    wandb_config['alpha'] = alpha
    wandb.init(project="step-distance-encoder", config=wandb_config)

    for epoch in range(num_epochs):
        state_encoder.train()
        total_loss = 0
        
        for i, batch in enumerate(train_loader):
            state = batch['state'].to(device)
            target_state = batch['target_state'].to(device)
            step_distance = batch['step_distance_to_target'].to(device)
            shape_w = batch['shape_w'].to(device)
            shape_h = batch['shape_h'].to(device)
            num_colors_grid = batch['num_colors_grid'].to(device)
            most_present_color = batch['most_present_color'].to(device)
            least_present_color = batch['least_present_color'].to(device)

            # Encode both states
            state_encoding = state_encoder(
                state,
                shape_w=shape_w,
                shape_h=shape_h,
                num_unique_colors=num_colors_grid,
                most_common_color=most_present_color,
                least_common_color=least_present_color
            )
            
            target_encoding = state_encoder(
                target_state,
                shape_w=shape_w,
                shape_h=shape_h,
                num_unique_colors=num_colors_grid,
                most_common_color=most_present_color,
                least_common_color=least_present_color
            )

            loss, _ = step_distance_loss(state_encoding, target_encoding, step_distance, alpha)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(state_encoder.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * state.size(0)

            if (i + 1) % log_interval == 0:
                print(f"\rEpoch {epoch+1} Batch {i+1}/{len(train_loader)} Loss: {loss.item():.4f}", end='', flush=True)

        avg_loss = total_loss / len(train_loader.dataset)
        val_loss, val_metrics = evaluate(state_encoder, val_loader, device, alpha)
        
        print(f"\rEpoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}", end='', flush=True)
        
        wandb.log({
            "train_loss": avg_loss,
            "val_loss": val_loss,
            "val_cosine_similarity": val_metrics['cosine_similarity'],
            "val_expected_similarity": val_metrics['expected_similarity'],
            "val_step_distance": val_metrics['step_distance']
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'state_encoder': state_encoder.state_dict(),
                'alpha': alpha,
                'config': config
            }, save_path)
            print(f"\nNew best model saved to {save_path}")
        else:
            epochs_no_improve += 1
            print(f"\nNo improvement for {epochs_no_improve} epoch(s)")
            
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1} due to no improvement in validation loss for {patience} epochs.")
            break

    wandb.finish()

if __name__ == "__main__":
    train_step_distance_encoder()
