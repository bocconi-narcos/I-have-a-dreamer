import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
import wandb
from src.models.state_encoder import StateEncoder
from src.models.predictors.reward_predictor import RewardPredictor
from src.data.replay_buffer_dataset import ReplayBufferDataset
import torch.nn.functional as F
import subprocess
from tqdm import tqdm

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def evaluate_reward_predictor(reward_predictor, state_encoder, target_encoder, dataloader, device, reward_criterion):
    """Evaluate the reward predictor on validation data."""
    reward_predictor.eval()
    state_encoder.eval()
    target_encoder.eval()
    
    total_reward_loss = 0
    total_reward_mae = 0
    total_reward_mse = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in dataloader:
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

            # Encode all three states with proper parameters
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
            
            # Compute losses
            reward_loss = reward_criterion(pred_reward.squeeze(-1), reward)
            reward_mae = F.l1_loss(pred_reward.squeeze(-1), reward)
            reward_mse = F.mse_loss(pred_reward.squeeze(-1), reward)
            
            total_reward_loss += reward_loss.item() * state.size(0)
            total_reward_mae += reward_mae.item() * state.size(0)
            total_reward_mse += reward_mse.item() * state.size(0)
            total_samples += state.size(0)

    avg_reward_loss = total_reward_loss / total_samples
    avg_reward_mae = total_reward_mae / total_samples
    avg_reward_mse = total_reward_mse / total_samples
    
    return avg_reward_loss, avg_reward_mae, avg_reward_mse

def train_reward_predictor():
    """
    Train the reward predictor to predict rewards based on current state, next state, and target state.
    The reward predictor takes three encoded states and outputs a scalar reward prediction.
    """
    config = load_config()
    
    # Initialize wandb
    wandb_config = config.copy()
    wandb.init(project="reward-predictor", config=wandb_config)
    
    # Buffer setup with fast tensor mode
    buffer_path = config['buffer_path']
    fast_buffer_path = buffer_path + '.fast.pt'
    if not os.path.exists(fast_buffer_path):
        print(f"Fast buffer {fast_buffer_path} not found. Preprocessing...")
        subprocess.run(['python', 'scripts/preprocess_buffer.py', buffer_path, fast_buffer_path], check=True)
    else:
        print(f"Using fast buffer: {fast_buffer_path}")
    
    # Model parameters
    encoder_type = config['encoder_type']
    latent_dim = config['latent_dim']
    encoder_params = config['encoder_params']
    num_color_selection_fns = config['action_embedders']['action_color_embedder']['num_actions']
    num_selection_fns = config['action_embedders']['action_selection_embedder']['num_actions']
    num_transform_actions = config['action_embedders']['action_transform_embedder']['num_actions']
    num_arc_colors = config['num_arc_colors']
    
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
    dataset = ReplayBufferDataset(
        buffer_path=fast_buffer_path,
        num_color_selection_fns=num_color_selection_fns,
        num_selection_fns=num_selection_fns,
        num_transform_actions=num_transform_actions,
        num_arc_colors=num_arc_colors,
        state_shape=state_shape,
        mode='end_to_end'  # Need next_state for reward prediction
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
            state_encoder.load_state_dict(checkpoint['state_encoder'])
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

    # Target encoder (EMA)
    target_encoder = copy.deepcopy(state_encoder)
    target_encoder.eval()
    for p in target_encoder.parameters():
        p.requires_grad = False

    # Create reward predictor
    reward_predictor = RewardPredictor(
        latent_dim=latent_dim,
        hidden_dim=config['reward_predictor'].get('hidden_dim', 128),
        transformer_depth=config['reward_predictor'].get('transformer_depth', 2),
        transformer_heads=config['reward_predictor'].get('transformer_heads', 2),
        transformer_dim_head=config['reward_predictor'].get('transformer_dim_head', 64),
        transformer_mlp_dim=config['reward_predictor'].get('transformer_mlp_dim', 128),
        dropout=config['reward_predictor'].get('transformer_dropout', 0.1),
        proj_dim=config['reward_predictor'].get('proj_dim', None)
    ).to(device)
    print(f"[RewardPredictor] Number of parameters: {sum(p.numel() for p in reward_predictor.parameters())}")

    # Loss function
    reward_criterion = nn.L1Loss()
    
    # Optimizer
    if use_pretrained_encoder and freeze_pretrained_encoder:
        optimizer = optim.AdamW(
            list(reward_predictor.parameters()), 
            lr=learning_rate
        )
    else:
        optimizer = optim.AdamW(
            list(state_encoder.parameters()) + list(reward_predictor.parameters()), 
            lr=learning_rate
        )

    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10
    save_path = 'best_model_reward_predictor.pth'
    
    print(f"Starting training with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    
    for epoch in range(num_epochs):
        state_encoder.train()
        reward_predictor.train()
        total_reward_loss = 0
        total_reward_mae = 0
        total_reward_mse = 0
        total_samples = 0
        
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)):
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

            # Encode all three states with proper parameters
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
            
            # Compute loss
            reward_loss = reward_criterion(pred_reward.squeeze(-1), reward)
            reward_mae = F.l1_loss(pred_reward.squeeze(-1), reward)
            reward_mse = F.mse_loss(pred_reward.squeeze(-1), reward)

            # Backward pass
            optimizer.zero_grad()
            reward_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(state_encoder.parameters()) + list(reward_predictor.parameters()),
                max_norm=1.0
            )
            optimizer.step()

            # EMA update for target encoder
            with torch.no_grad():
                for target_param, source_param in zip(target_encoder.parameters(), state_encoder.parameters()):
                    target_param.data.mul_(0.995).add_(source_param.data, alpha=1 - 0.995)

            # Accumulate metrics
            total_reward_loss += reward_loss.item() * state.size(0)
            total_reward_mae += reward_mae.item() * state.size(0)
            total_reward_mse += reward_mse.item() * state.size(0)
            total_samples += state.size(0)

            # Log batch metrics
            if (i + 1) % log_interval == 0:
                wandb.log({
                    "batch_reward_loss": reward_loss.item(),
                    "batch_reward_mae": reward_mae.item(),
                    "batch_reward_mse": reward_mse.item(),
                    "epoch": epoch + 1,
                    "batch": i + 1
                })

        # Compute average training metrics
        avg_reward_loss = total_reward_loss / total_samples
        avg_reward_mae = total_reward_mae / total_samples
        avg_reward_mse = total_reward_mse / total_samples

        # Evaluate on validation set
        val_reward_loss, val_reward_mae, val_reward_mse = evaluate_reward_predictor(
            reward_predictor, state_encoder, target_encoder, val_loader, device, reward_criterion
        )

        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train - Loss: {avg_reward_loss:.4f}, MAE: {avg_reward_mae:.4f}, MSE: {avg_reward_mse:.4f}")
        print(f"  Val   - Loss: {val_reward_loss:.4f}, MAE: {val_reward_mae:.4f}, MSE: {val_reward_mse:.4f}")

        # Log to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_reward_loss": avg_reward_loss,
            "train_reward_mae": avg_reward_mae,
            "train_reward_mse": avg_reward_mse,
            "val_reward_loss": val_reward_loss,
            "val_reward_mae": val_reward_mae,
            "val_reward_mse": val_reward_mse,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Save best model
        if val_reward_loss < best_val_loss:
            best_val_loss = val_reward_loss
            epochs_no_improve = 0
            torch.save({
                'state_encoder': state_encoder.state_dict(),
                'reward_predictor': reward_predictor.state_dict(),
                'target_encoder': target_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss
            }, save_path)
            print(f"New best model saved to {save_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
            
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss for {patience} epochs.")
            break

    wandb.finish()
    print("Training completed!")

if __name__ == "__main__":
    train_reward_predictor()
