#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
import wandb
from src.data.replay_buffer_dataset import ReplayBufferDataset
from src.models.state_encoder import StateEncoder
import torch.nn.functional as F
from tqdm import tqdm

# --- R² calculation function ---
def calculate_r2_score(y_true, y_pred):
    """
    Calculate R-squared (R²) score.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        R² score (float)
    """
    # Calculate mean of true values
    y_mean = torch.mean(y_true)
    
    # Calculate total sum of squares (TSS)
    tss = torch.sum((y_true - y_mean) ** 2)
    
    # Calculate residual sum of squares (RSS)
    rss = torch.sum((y_true - y_pred) ** 2)
    
    # Debug prints
    # print(f"R² Debug - y_true mean: {y_mean.item():.4f}")
    # print(f"R² Debug - TSS: {tss.item():.4f}")
    # print(f"R² Debug - RSS: {rss.item():.4f}")
    
    # Handle edge cases
    if tss == 0:
        # If TSS is 0, all true values are the same
        if rss == 0:
            return 1.0  # Perfect prediction
        else:
            return 0.0  # No predictive power
    
    # Calculate R²
    r2 = 1 - (rss / tss)
    
    # Handle numerical issues
    if torch.isnan(r2) or torch.isinf(r2):
        # print(f"R² Debug - NaN or Inf detected: {r2}")
        return 0.0
    
    # print(f"R² Debug - Final R²: {r2.item():.4f}")
    return r2.item()

class StepDistanceDataset(ReplayBufferDataset):
    """Wrapper around ReplayBufferDataset that includes step_distance_to_target field"""
    
    def __init__(self, buffer_path, num_color_selection_fns, num_selection_fns, 
                 num_transform_actions, num_arc_colors, state_shape, mode='end_to_end', num_samples=None):
        super().__init__(buffer_path, num_color_selection_fns, num_selection_fns, 
                        num_transform_actions, num_arc_colors, state_shape, mode, num_samples)
        
        # Load the original buffer to get step_distance_to_target
        if os.path.exists(buffer_path):
            original_buffer = torch.load(buffer_path, map_location='cpu')
            if isinstance(original_buffer, dict) and 'step_distance_to_target' in original_buffer:
                self.step_distances = original_buffer['step_distance_to_target']
            else:
                self.step_distances = None
        else:
            self.step_distances = None
    
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        
        # Add step_distance_to_target from the original buffer
        if self.step_distances is not None:
            sample['step_distance_to_target'] = self.step_distances[idx].detach().clone().float()
        else:
            # Fallback: compute simple distance
            state = sample['state']
            target_state = sample['target_state']
            diff = (state != target_state).float()
            sample['step_distance_to_target'] = diff.sum()
        
        return sample

class StepDistanceMLP(nn.Module):
    def __init__(self, latent_dim):
        super(StepDistanceMLP, self).__init__()
        input_dim = 2 * latent_dim  # concatenated state and target encodings
        hidden_dim = 64
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state_encoding, target_encoding):
        # Concatenate state and target encodings
        combined = torch.cat([state_encoding, target_encoding], dim=-1)
        return self.mlp(combined)

def train_step_distance_mlp():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wandb.init(project="step-distance-mlp", config=config)
    
    # Get required parameters from config
    buffer_path = config['buffer_path']
    encoder_params = config['encoder_params']
    image_size = encoder_params.get('image_size', [10, 10])
    input_channels = encoder_params.get('input_channels', 1)
    
    if isinstance(image_size, int):
        state_shape = (input_channels, image_size, image_size)
    else:
        state_shape = (input_channels, image_size[0], image_size[1])
    
    # Use the wrapper dataset that includes step_distance_to_target
    dataset = StepDistanceDataset(
        buffer_path=buffer_path,
        num_color_selection_fns=config['action_embedders']['action_color_embedder']['num_actions'],
        num_selection_fns=config['action_embedders']['action_selection_embedder']['num_actions'],
        num_transform_actions=config['action_embedders']['action_transform_embedder']['num_actions'],
        num_arc_colors=config['num_arc_colors'],
        state_shape=state_shape,
        mode='end_to_end'
    )
    
    train_size = int(0.8 * len(dataset))
    train_dataset, val_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])
    
    # Models
    state_encoder = StateEncoder(
        image_size=image_size,
        input_channels=input_channels,
        latent_dim=config['latent_dim'],
        encoder_params=config['encoder_params']
    ).to(device)
    
    step_distance_mlp = StepDistanceMLP(config['latent_dim']).to(device)
    
    # Load pretrained encoder if available
    if config.get('use_pretrained_encoder', False):
        pretrained_path = config.get('pretrained_encoder_path', 'weights/best_model_state_encoder.pth')
        try:
            checkpoint = torch.load(pretrained_path, map_location=device)
            state_encoder.load_state_dict(checkpoint.get('state_encoder', checkpoint))
        except:
            pass
    
    # Optimizer
    optimizer = optim.Adam(list(step_distance_mlp.parameters()) + list(state_encoder.parameters()), lr=config['learning_rate'])
    
    # Training
    # print("Starting training loop...")
    for epoch in range(config['num_epochs']):
        # print(f"Epoch {epoch + 1}/{config['num_epochs']}")
        state_encoder.train()
        step_distance_mlp.train()
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config['num_epochs']}")):
            
            state = batch['state'].to(device)
            target_state = batch['target_state'].to(device)
            step_distance = batch['step_distance_to_target'].to(device)
            
            # print(f"    State shape: {state.shape}")
            # print(f"    Target state shape: {target_state.shape}")
            # print(f"    Step distance shape: {step_distance.shape}")
            
            # Get additional required arguments for StateEncoder
            shape_h = batch['shape_h'].to(device)
            shape_w = batch['shape_w'].to(device)
            most_present_color = batch['most_present_color'].to(device)
            least_present_color = batch['least_present_color'].to(device)
            num_colors_grid = batch['num_colors_grid'].to(device)
            
            # print(f"    Encoding states...")
            # Encode states
            state_encoding = state_encoder(state, shape_h, shape_w, most_present_color, least_present_color, num_colors_grid)
            target_encoding = state_encoder(target_state, shape_h, shape_w, most_present_color, least_present_color, num_colors_grid)
            
            # print(f"    State encoding shape: {state_encoding.shape}")
            # print(f"    Target encoding shape: {target_encoding.shape}")
            
            # print(f"    Predicting step distance...")
            # Predict step distance
            predicted_distance = step_distance_mlp(state_encoding, target_encoding)
            
            # print(f"    Predicted distance shape: {predicted_distance.shape}")
            
            # Calculate loss
            loss_mse = F.mse_loss(predicted_distance.squeeze(-1), step_distance)
            loss_mae = F.l1_loss(predicted_distance.squeeze(-1), step_distance)
            
            # Combine losses for single backward pass
            total_loss = loss_mse + loss_mae
            
            # print(f"    Loss: {loss.item():.4f}")
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # print(f"    Step completed!")
            
            if i % config['log_interval'] == 0:
                # Calculate validation metrics for logging
                state_encoder.eval()
                step_distance_mlp.eval()
                val_loss_mse = 0
                val_loss_mae = 0
                val_r2 = 0
                val_samples = 0
                
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_state = val_batch['state'].to(device)
                        val_target_state = val_batch['target_state'].to(device)
                        val_step_distance = val_batch['step_distance_to_target'].to(device)
                        
                        # Get additional required arguments for StateEncoder
                        val_shape_h = val_batch['shape_h'].to(device)
                        val_shape_w = val_batch['shape_w'].to(device)
                        val_most_present_color = val_batch['most_present_color'].to(device)
                        val_least_present_color = val_batch['least_present_color'].to(device)
                        val_num_colors_grid = val_batch['num_colors_grid'].to(device)
                        
                        # Encode states
                        val_state_encoding = state_encoder(val_state, val_shape_h, val_shape_w, val_most_present_color, val_least_present_color, val_num_colors_grid)
                        val_target_encoding = state_encoder(val_target_state, val_shape_h, val_shape_w, val_most_present_color, val_least_present_color, val_num_colors_grid)
                        
                        # Predict step distance
                        val_predicted_distance = step_distance_mlp(val_state_encoding, val_target_encoding)
                        
                        # Calculate metrics
                        val_loss_mse += F.mse_loss(val_predicted_distance.squeeze(-1), val_step_distance).item()
                        val_loss_mae += F.l1_loss(val_predicted_distance.squeeze(-1), val_step_distance).item()
                        val_r2 += calculate_r2_score(val_step_distance, val_predicted_distance.squeeze(-1))
                        val_samples += 1
                        
                        # Only calculate for first few validation batches to avoid slowing down training
                        if val_samples >= 5:
                            break
                
                val_loss_mse /= val_samples
                val_loss_mae /= val_samples
                val_r2 /= val_samples
                
                # Reset to training mode
                state_encoder.train()
                step_distance_mlp.train()
                
                wandb.log({
                    'epoch': epoch + 1,
                    'batch': i + 1,
                    'train_loss_mse': loss_mse.item(),
                    'train_loss_mae': loss_mae.item(),
                    'val_loss_mse': val_loss_mse,
                    'val_loss_mae': val_loss_mae,
                    'train_r2': calculate_r2_score(step_distance, predicted_distance.squeeze(-1)),
                    'val_r2': val_r2,
                })
        
        # print(f"Epoch {epoch + 1} completed!")
        
        # Validation
        # print("Starting validation...")
        state_encoder.eval()
        step_distance_mlp.eval()
        val_loss = 0
        val_r2 = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(val_loader, desc="Validation")):
                # print(f"  Validation batch {batch_idx + 1}/{len(val_loader)}")
                
                state = batch['state'].to(device)
                target_state = batch['target_state'].to(device)
                step_distance = batch['step_distance_to_target'].to(device)
                
                # Get additional required arguments for StateEncoder
                shape_h = batch['shape_h'].to(device)
                shape_w = batch['shape_w'].to(device)
                most_present_color = batch['most_present_color'].to(device)
                least_present_color = batch['least_present_color'].to(device)
                num_colors_grid = batch['num_colors_grid'].to(device)
                
                # Encode states
                state_encoding = state_encoder(state, shape_h, shape_w, most_present_color, least_present_color, num_colors_grid)
                target_encoding = state_encoder(target_state, shape_h, shape_w, most_present_color, least_present_color, num_colors_grid)
                
                # Predict step distance
                predicted_distance = step_distance_mlp(state_encoding, target_encoding)
                
                # Calculate metrics
                val_loss += F.mse_loss(predicted_distance.squeeze(-1), step_distance).item()
                val_r2 += calculate_r2_score(step_distance, predicted_distance.squeeze(-1))
        
        val_loss /= len(val_loader)
        val_r2 /= len(val_loader)
        
        print(f"Validation - Loss: {val_loss:.4f}, R2: {val_r2:.4f}")
        
        wandb.log({
            'epoch': epoch + 1,
            'val_loss': val_loss,
            'val_r2': val_r2
        })
    
    wandb.finish()

if __name__ == "__main__":
    train_step_distance_mlp() 