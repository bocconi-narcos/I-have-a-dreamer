#!/usr/bin/env python3
"""
Train Step Distance Predictor

This script trains a state encoder to learn representations where the cosine similarity
between state encodings correlates with the step distance between states. The training
objective uses an exponential decay function: similarity = exp(-step_distance).

Key features:
- Uses the same state encoder architecture as other predictors
- Loads step_distance_to_target from the dataset
- Calculates cosine similarity between current and target state encodings
- Uses MAE loss: MAE(cosine_similarity, exp(-step_distance))
- Includes comprehensive logging and validation with W&B
- Implements early stopping based on validation loss

Usage:
    python train_step_distance_predictor.py

The script will automatically load configuration from config.yaml and
use the replay buffer specified in the config.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
import wandb
from src.models.state_encoder import StateEncoder
from src.data.replay_buffer_dataset import ReplayBufferDataset
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class StepDistanceDataset(ReplayBufferDataset):
    """Extended dataset that includes target_state and step_distance_to_target"""
    
    def __init__(self, buffer_path, num_color_selection_fns, num_selection_fns, 
                 num_transform_actions, num_arc_colors, state_shape, mode='full', num_samples=None):
        # Initialize base class attributes without calling parent __init__
        self.buffer_path = buffer_path
        self.num_color_selection_fns = num_color_selection_fns
        self.num_selection_fns = num_selection_fns
        self.num_transform_actions = num_transform_actions
        self.num_arc_colors = num_arc_colors
        self.state_shape = state_shape
        self.mode = mode
        
        # Load buffer data specifically for step distance training
        if os.path.exists(buffer_path):
            print(f"Loading replay buffer from {buffer_path}")
            if buffer_path.endswith('.pt'):
                import time
                start_time = time.time()
                self.buffer = self._load_pt_buffer(buffer_path)
                end_time = time.time()
                print(f"Loaded {len(self.buffer)} transitions in {end_time - start_time:.2f} seconds")
            else:
                raise ValueError(f"Unsupported buffer file format: {buffer_path}. Please use .pt files.")
        else:
            print(f"ERROR: Buffer file {buffer_path} not found. Please provide a valid replay buffer file.")
            raise FileNotFoundError(f"Buffer file {buffer_path} not found.")
        
        # Limit samples if specified (for testing)
        if num_samples is not None:
            self.buffer = self.buffer[:num_samples]
        
        print(f"Dataset initialized with {len(self.buffer)} samples in {mode} mode")
    
    def _load_pt_buffer(self, buffer_path):
        """Load replay buffer data from a .pt file including target state and step distance."""
        buffer = []
        
        # Load the buffer data
        buffer_dict = torch.load(buffer_path, map_location='cpu')
        
        # Handle both dictionary format and list format
        if isinstance(buffer_dict, dict) and 'state' in buffer_dict:
            # Dictionary format with arrays for each field
            num_transitions = len(buffer_dict['state'])
            
            for i in range(num_transitions):
                # Skip transitions with invalid step distance (-1)
                step_distance = buffer_dict['step_distance_to_target'][i]
                if step_distance < 0:
                    continue
                    
                transition = {
                    'state': buffer_dict['state'][i],
                    'action': {
                        'colour': buffer_dict['action_colour'][i],
                        'selection': buffer_dict['action_selection'][i],
                        'transform': buffer_dict['action_transform'][i]
                    },
                    'selection_mask': buffer_dict['selection_mask'][i],
                    'next_state': buffer_dict['next_state'][i],
                    'target_state': buffer_dict['target_state'][i],
                    'step_distance_to_target': step_distance,
                    'colour': buffer_dict['colour'][i],
                    'reward': buffer_dict['reward'][i],
                    'done': buffer_dict['done'][i],
                    'transition_type': buffer_dict['transition_type'][i],
                    'shape_h': buffer_dict['shape_h'][i],
                    'shape_w': buffer_dict['shape_w'][i],
                    'num_colors_grid': buffer_dict['num_colors_grid'][i],
                    'most_present_color': buffer_dict['most_present_color'][i],
                    'least_present_color': buffer_dict['least_present_color'][i]
                }
                buffer.append(transition)
        else:
            # List format - filter out transitions with invalid step distance
            for transition in buffer_dict:
                if transition.get('step_distance_to_target', -1) >= 0:
                    buffer.append(transition)
        
        return buffer
    
    def __len__(self):
        return len(self.buffer)
    
    def _to_tensor(self, data, dtype):
        """Convert data to tensor, handling both tensor and non-tensor inputs."""
        if torch.is_tensor(data):
            return data.clone().detach().to(dtype)
        else:
            return torch.tensor(data, dtype=dtype)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset including target state and step distance."""
        transition = self.buffer[idx]
        
        # Extract states and convert to tensor
        state = self._to_tensor(transition['state'], torch.long)
        target_state = self._to_tensor(transition['target_state'], torch.long)
        step_distance = self._to_tensor(transition['step_distance_to_target'], torch.float32)
        
        # Extract grid statistics for state
        shape_h = self._to_tensor(transition['shape_h'], torch.long)
        shape_w = self._to_tensor(transition['shape_w'], torch.long)
        num_colors_grid = self._to_tensor(transition['num_colors_grid'], torch.long)
        most_present_color = self._to_tensor(transition['most_present_color'], torch.long)
        least_present_color = self._to_tensor(transition['least_present_color'], torch.long)
        
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

class StepDistanceMLP(nn.Module):
    """
    Simple MLP that predicts step distance from encoded state and encoded target.
    """
    
    def __init__(self, latent_dim):
        """
        Initialize the simple StepDistanceMLP.
        
        Args:
            latent_dim: Dimension of the encoded state/target representations
        """
        super(StepDistanceMLP, self).__init__()
        
        # Simple 2-layer MLP: input -> hidden -> output
        input_dim = 2 * latent_dim  # concatenated state and target encodings
        hidden_dim = 64
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state_encoding, target_encoding):
        """
        Forward pass to predict step distance.
        
        Args:
            state_encoding: Encoded state representation [B, latent_dim]
            target_encoding: Encoded target state representation [B, latent_dim]
        
        Returns:
            predicted_distance: Predicted step distance [B, 1]
        """
        # Concatenate state and target encodings
        combined = torch.cat([state_encoding, target_encoding], dim=-1)
        
        # Predict step distance
        predicted_distance = self.mlp(combined)
        
        # Ensure non-negative output
        predicted_distance = F.relu(predicted_distance)
        
        return predicted_distance

def step_distance_mlp_loss(predicted_distance, actual_distance):
    """
    Compute loss for the step distance MLP.
    
    Args:
        predicted_distance: Predicted step distance from MLP [B, 1]
        actual_distance: Actual step distance from dataset [B]
    
    Returns:
        loss: MAE loss between predicted and actual step distance
        metrics: Dictionary with loss components for monitoring
    """
    
    # Squeeze predicted_distance to match actual_distance shape
    predicted_distance = predicted_distance.squeeze(-1)
    
    # MAE loss
    loss = F.l1_loss(predicted_distance, actual_distance)
    
    # Additional metrics for monitoring
    mse_loss = F.mse_loss(predicted_distance, actual_distance)
    
    r2_score = r2_score(predicted_distance, actual_distance)
    
    metrics = {
        'mae_loss': loss.item(),
        'mse_loss': mse_loss.item(),
    }
    
    return loss, metrics

def step_distance_loss(state_encoding, target_encoding, step_distance):
    """
    Compute step distance loss based on cosine similarity and expected similarity.
    
    This loss function trains the encoder to produce representations where:
    - States that are 0 steps apart (identical) have cosine similarity = 1
    - States that are many steps apart have cosine similarity approaching 0
    - The decay follows an exponential: exp(-step_distance)
    
    Args:
        state_encoding: Encoded state representation [B, latent_dim]
        target_encoding: Encoded target state representation [B, latent_dim]
        step_distance: Step distance to target [B] (non-negative values)
    
    Returns:
        loss: MAE loss between cosine similarity and expected similarity
        metrics: Dictionary with loss components for monitoring
    """
    # Normalize encodings for cosine similarity
    state_norm = F.normalize(state_encoding, p=2, dim=-1)
    target_norm = F.normalize(target_encoding, p=2, dim=-1)
    
    # Compute cosine similarity
    cosine_sim = (state_norm * target_norm).sum(dim=-1)  # [B]
    
    # Expected similarity: exp(-step_distance)
    # When step_distance = 0, exp(-0) = 1 (perfect similarity)
    # When step_distance -> inf, exp(-inf) -> 0 (no similarity)
    expected_sim = torch.exp(-step_distance)
    
    # MAE loss between actual and expected similarity
    loss = F.l1_loss(cosine_sim, expected_sim) #NOTE: use it if want to use MAE loss
    # loss = F.huber_loss(cosine_sim, expected_sim) #NOTE: use it if want to use Huber loss
    
    metrics = {
        'cosine_similarity': cosine_sim.mean().item(),
        'expected_similarity': expected_sim.mean().item(),
        'step_distance': step_distance.mean().item(),
        'mae_loss': loss.item()
    }
    
    return loss, metrics

def evaluate(encoder, dataloader, device):
    """Evaluate the encoder on validation data."""
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
            
            loss, metrics = step_distance_loss(state_encoding, target_encoding, step_distance)
            
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

def train_step_distance_predictor():
    """
    Train the step distance predictor.
    
    The training process:
    1. Load current state and target state from dataset
    2. Encode both states using StateEncoder
    3. Calculate cosine similarity between encodings
    4. Use MAE loss: MAE(cosine_similarity, exp(-step_distance))
    5. Train the encoder to learn distance-aware representations
    """
    config = load_config()
    
    # Initialize wandb
    wandb_config = config.copy()
    wandb_available = True
    try:
        wandb.init(project="step-distance-predictor", config=wandb_config, settings=wandb.Settings(init_timeout=180))
        print("Wandb initialized successfully!")
    except Exception as e:
        print(f"Wandb initialization failed: {e}")
        print("Continuing without wandb logging...")
        wandb_available = False
    
    # Buffer setup
    buffer_path = config['buffer_path']
    print(f"Using buffer: {buffer_path}")
    
    # Model parameters
    encoder_type = config['encoder_type']
    latent_dim = config['latent_dim']
    encoder_params = config['encoder_params']
    
    # Training parameters
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    num_workers = config['num_workers']
    log_interval = config['log_interval']
    
    # Simple stabilization parameters
    gradient_clip_norm = config.get('gradient_clip_norm', 1.0)
    patience = config.get('early_stopping_patience', 10)  # Early stopping patience

    # State shape
    image_size = encoder_params.get('image_size', [10, 10])
    input_channels = encoder_params.get('input_channels', 1)
    if isinstance(image_size, int):
        state_shape = (input_channels, image_size, image_size)
    else:
        state_shape = (input_channels, image_size[0], image_size[1])

    # Dataset setup
    dataset = StepDistanceDataset(
        buffer_path=buffer_path,
        num_color_selection_fns=config['action_embedders']['action_color_embedder']['num_actions'],
        num_selection_fns=config['action_embedders']['action_selection_embedder']['num_actions'],
        num_transform_actions=config['action_embedders']['action_transform_embedder']['num_actions'],
        num_arc_colors=config['num_arc_colors'],
        state_shape=state_shape,
        mode='full'
    )
    
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Device selection
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
        print('Using device: MPS (Apple Silicon GPU)')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using device: CUDA')
    else:
        device = torch.device('cpu')
        print('Using device: CPU')

    # Load pretrained encoder if specified
    use_pretrained_encoder = config.get('use_pretrained_encoder', False)
    pretrained_encoder_path = config.get('pretrained_encoder_path', 'best_model_autoencoder.pth')
    freeze_pretrained_encoder = config.get('freeze_pretrained_encoder', False)

    # Create state encoder
    state_encoder = StateEncoder(
        image_size=image_size,
        input_channels=input_channels,
        latent_dim=latent_dim,
        encoder_params=encoder_params
    ).to(device)

    if use_pretrained_encoder:
        if os.path.exists(pretrained_encoder_path):
            print(f"Loading pretrained encoder from {pretrained_encoder_path}")
            checkpoint = torch.load(pretrained_encoder_path, map_location=device)
            # Handle different checkpoint structures
            if 'state_encoder' in checkpoint:
                # Checkpoint has state_encoder wrapped in a dictionary
                state_encoder.load_state_dict(checkpoint['state_encoder'])
            else:
                # Checkpoint contains the state dict directly
                state_encoder.load_state_dict(checkpoint)
            print("Pretrained encoder loaded successfully!")
            if freeze_pretrained_encoder:
                for param in state_encoder.parameters():
                    param.requires_grad = False
                print("Encoder parameters frozen.")
            else:
                print("Encoder parameters will be fine-tuned.")
        else:
            print(f"Warning: Pretrained encoder path {pretrained_encoder_path} not found. Training from scratch.")
    else:
        print("Training encoder from scratch.")

    # Optimizer
    if use_pretrained_encoder and freeze_pretrained_encoder:
        print("Warning: Encoder is frozen, but no other parameters to optimize.")
        optimizer = optim.AdamW([], lr=learning_rate)  # Empty parameter list
    else:
        optimizer = optim.AdamW(state_encoder.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    save_path = os.path.join('weights', 'best_model_step_distance_encoder.pth')
    os.makedirs('weights', exist_ok=True)
    
    print(f"Starting training with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    print(f"Features:")
    print(f"  - Step distance prediction with cosine similarity")
    print(f"  - MAE loss: MAE(cosine_similarity, exp(-step_distance))")
    print(f"  - Gradient clipping: {gradient_clip_norm}")
    print(f"  - Early stopping patience: {patience}")
    
    # Track global step for proper logging
    global_step = 0
    
    for epoch in range(num_epochs):
        state_encoder.train()
        total_loss = 0
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)):
            # Step 1: Load current state and target state
            state = batch['state'].to(device)
            target_state = batch['target_state'].to(device)
            step_distance = batch['step_distance_to_target'].to(device)
            
            # Extract metadata
            shape_h = batch['shape_h'].to(device)
            shape_w = batch['shape_w'].to(device)
            num_colors_grid = batch['num_colors_grid'].to(device)
            most_present_color = batch['most_present_color'].to(device)
            least_present_color = batch['least_present_color'].to(device)

            # Step 2: Encode both states
            state_encoding = state_encoder(
                state,
                shape_h=shape_h,
                shape_w=shape_w,
                num_unique_colors=num_colors_grid,
                most_common_color=most_present_color,
                least_common_color=least_present_color
            )
            
            target_encoding = state_encoder(
                target_state,
                shape_h=shape_h,
                shape_w=shape_w,
                num_unique_colors=num_colors_grid,
                most_common_color=most_present_color,
                least_common_color=least_present_color
            )

            # Step 3 & 4: Calculate loss (cosine similarity vs exp(-step_distance))
            loss, metrics = step_distance_loss(state_encoding, target_encoding, step_distance)

            # Backward pass
            if not (use_pretrained_encoder and freeze_pretrained_encoder):
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(state_encoder.parameters(), max_norm=gradient_clip_norm)
                optimizer.step()

            total_loss += loss.item() * state.size(0)
            global_step += 1

            # Log training metrics every log_interval steps
            if global_step % log_interval == 0 and wandb_available:
                wandb.log({
                    "step": global_step,
                    "epoch": epoch + 1,
                    "batch_loss": loss.item(),
                    "batch_cosine_similarity": metrics['cosine_similarity'],
                    "batch_expected_similarity": metrics['expected_similarity'],
                    "batch_step_distance": metrics['step_distance'],
                })

        avg_loss = total_loss / len(train_dataset)
        val_loss, val_metrics = evaluate(state_encoder, val_loader, device)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Val Cosine Sim: {val_metrics['cosine_similarity']:.4f} | Expected Sim: {val_metrics['expected_similarity']:.4f} | Step Distance: {val_metrics['step_distance']:.4f}")
        
        # Log to wandb
        if wandb_available:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_loss": val_loss,
                "val_cosine_similarity": val_metrics['cosine_similarity'],
                "val_expected_similarity": val_metrics['expected_similarity'],
                "val_step_distance": val_metrics['step_distance']
            })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'state_encoder': state_encoder.state_dict(),
                'config': config
            }, save_path)
            print(f"New best model saved to {save_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
            
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss for {patience} epochs.")
            break

    if wandb_available:
        wandb.finish()
    
    print("Training completed!")

def train_step_distance_mlp():
    """
    Train the simple StepDistanceMLP to predict step distance.
    """
    config = load_config()
    
    # Initialize wandb
    wandb_config = config.copy()
    wandb_available = True
    try:
        wandb.init(project="step-distance-mlp", config=wandb_config, settings=wandb.Settings(init_timeout=180))
        print("Wandb initialized successfully!")
    except Exception as e:
        print(f"Wandb initialization failed: {e}")
        print("Continuing without wandb logging...")
        wandb_available = False
    
    # Buffer setup
    buffer_path = config['buffer_path']
    print(f"Using buffer: {buffer_path}")
    
    # Model parameters
    encoder_type = config['encoder_type']
    latent_dim = config['latent_dim']
    encoder_params = config['encoder_params']
    
    # Training parameters
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    num_workers = config['num_workers']
    log_interval = config['log_interval']
    
    # State shape
    image_size = encoder_params.get('image_size', [10, 10])
    input_channels = encoder_params.get('input_channels', 1)
    if isinstance(image_size, int):
        state_shape = (input_channels, image_size, image_size)
    else:
        state_shape = (input_channels, image_size[0], image_size[1])

    # Dataset setup
    dataset = StepDistanceDataset(
        buffer_path=buffer_path,
        num_color_selection_fns=config['action_embedders']['action_color_embedder']['num_actions'],
        num_selection_fns=config['action_embedders']['action_selection_embedder']['num_actions'],
        num_transform_actions=config['action_embedders']['action_transform_embedder']['num_actions'],
        num_arc_colors=config['num_arc_colors'],
        state_shape=state_shape,
        mode='full'
    )
    
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Device selection
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
        print('Using device: MPS (Apple Silicon GPU)')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using device: CUDA')
    else:
        device = torch.device('cpu')
        print('Using device: CPU')

    # Create state encoder (frozen)
    state_encoder = StateEncoder(
        image_size=image_size,
        input_channels=input_channels,
        latent_dim=latent_dim,
        encoder_params=encoder_params
    ).to(device)
    
    # Load pretrained encoder
    pretrained_encoder_path = config.get('pretrained_encoder_path', 'weights/best_model_state_encoder.pth')
    if os.path.exists(pretrained_encoder_path):
        print(f"Loading pretrained encoder from {pretrained_encoder_path}")
        checkpoint = torch.load(pretrained_encoder_path, map_location=device)
        # Handle different checkpoint structures
        if 'state_encoder' in checkpoint:
            # Checkpoint has state_encoder wrapped in a dictionary
            state_encoder.load_state_dict(checkpoint['state_encoder'])
        else:
            # Checkpoint contains the state dict directly
            state_encoder.load_state_dict(checkpoint)
        print("Pretrained encoder loaded successfully!")
        # Freeze encoder
        for param in state_encoder.parameters():
            param.requires_grad = False
        print("Encoder parameters frozen.")
    else:
        print(f"Warning: Pretrained encoder path {pretrained_encoder_path} not found.")
        return

    # Create MLP
    mlp = StepDistanceMLP(latent_dim=latent_dim).to(device)
    print(f"Created MLP with {sum(p.numel() for p in mlp.parameters())} parameters")

    # Optimizer (only for MLP)
    optimizer = optim.AdamW(mlp.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = config.get('early_stopping_patience', 10)
    save_path = os.path.join('weights', 'best_model_step_distance_mlp.pth')
    os.makedirs('weights', exist_ok=True)
    
    print(f"Starting MLP training with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    
    for epoch in range(num_epochs):
        mlp.train()
        total_loss = 0
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)):
            state = batch['state'].to(device)
            target_state = batch['target_state'].to(device)
            step_distance = batch['step_distance_to_target'].to(device)
            
            # Extract metadata
            shape_h = batch['shape_h'].to(device)
            shape_w = batch['shape_w'].to(device)
            num_colors_grid = batch['num_colors_grid'].to(device)
            most_present_color = batch['most_present_color'].to(device)
            least_present_color = batch['least_present_color'].to(device)

            # Encode states (frozen encoder)
            with torch.no_grad():
                state_encoding = state_encoder(
                    state,
                    shape_h=shape_h,
                    shape_w=shape_w,
                    num_unique_colors=num_colors_grid,
                    most_common_color=most_present_color,
                    least_common_color=least_present_color
                )
                
                target_encoding = state_encoder(
                    target_state,
                    shape_h=shape_h,
                    shape_w=shape_w,
                    num_unique_colors=num_colors_grid,
                    most_common_color=most_present_color,
                    least_common_color=least_present_color
                )

            # Predict step distance
            predicted_distance = mlp(state_encoding, target_encoding)
            
            # Compute loss
            loss, metrics = step_distance_mlp_loss(predicted_distance, step_distance)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * state.size(0)

        avg_loss = total_loss / len(train_dataset)
        
        # Validation
        mlp.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                state = batch['state'].to(device)
                target_state = batch['target_state'].to(device)
                step_distance = batch['step_distance_to_target'].to(device)
                
                shape_h = batch['shape_h'].to(device)
                shape_w = batch['shape_w'].to(device)
                num_colors_grid = batch['num_colors_grid'].to(device)
                most_present_color = batch['most_present_color'].to(device)
                least_present_color = batch['least_present_color'].to(device)

                state_encoding = state_encoder(
                    state,
                    shape_h=shape_h,
                    shape_w=shape_w,
                    num_unique_colors=num_colors_grid,
                    most_common_color=most_present_color,
                    least_common_color=least_present_color
                )
                
                target_encoding = state_encoder(
                    target_state,
                    shape_h=shape_h,
                    shape_w=shape_w,
                    num_unique_colors=num_colors_grid,
                    most_common_color=most_present_color,
                    least_common_color=least_present_color
                )

                predicted_distance = mlp(state_encoding, target_encoding)
                loss, metrics = step_distance_mlp_loss(predicted_distance, step_distance)
                val_loss += loss.item() * state.size(0)
        
        val_loss = val_loss / len(val_dataset)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Log to wandb
        if wandb_available:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "val_loss": val_loss
            })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'mlp': mlp.state_dict(),
                'config': config
            }, save_path)
            print(f"New best model saved to {save_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
            
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if wandb_available:
        wandb.finish()
    
    print("MLP training completed!")

if __name__ == "__main__":
    # Choose which training function to run
    train_step_distance_predictor()  # Original cosine similarity approach
    # train_step_distance_mlp()  # New MLP approach
