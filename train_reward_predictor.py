import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
import yaml
import wandb
from src.models.state_encoder import StateEncoder
from src.models.predictors.reward_predictor import RewardPredictor, RewardPredictorLoss
from src.data.replay_buffer_dataset import ReplayBufferDataset
import torch.nn.functional as F
import subprocess
from tqdm import tqdm
import numpy as np

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

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
        return 0.0
    
    return r2.item()

def evaluate_reward_predictor(reward_predictor, state_encoder, target_encoder, dataloader, device, reward_criterion):
    """Evaluate the reward predictor on validation data."""
    reward_predictor.eval()
    state_encoder.eval()
    target_encoder.eval()
    
    total_reward_mse = 0
    total_reward_mae = 0
    total_samples = 0
    
    # For R² calculation, we need to collect all predictions and targets
    all_predictions = []
    all_targets = []
    all_uncertainties = []
    
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

            # Predict reward with new SOTA model
            pred_reward, uncertainty, contrastive_outputs = reward_predictor(latent_t, latent_tp1, latent_target)
            
            # Compute MSE and MAE losses
            reward_mse = F.mse_loss(pred_reward.squeeze(-1), reward)
            reward_mae = F.l1_loss(pred_reward.squeeze(-1), reward)
            
            # Collect predictions and targets for R² calculation
            all_predictions.append(pred_reward.squeeze(-1))
            all_targets.append(reward)
            if uncertainty is not None:
                all_uncertainties.append(uncertainty.squeeze(-1))
            
            # Accumulate metrics
            total_reward_mse += reward_mse.item() * state.size(0)
            total_reward_mae += reward_mae.item() * state.size(0)
            total_samples += state.size(0)
    
    # Calculate R² score
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    r2_score = calculate_r2_score(all_targets, all_predictions)
    
    # Calculate uncertainty statistics if available
    uncertainty_stats = {}
    if all_uncertainties:
        all_uncertainties = torch.cat(all_uncertainties, dim=0)
        uncertainty_stats = {
            'mean_uncertainty': all_uncertainties.mean().item(),
            'std_uncertainty': all_uncertainties.std().item(),
            'max_uncertainty': all_uncertainties.max().item(),
            'min_uncertainty': all_uncertainties.min().item()
        }
    
    # Debug: Print validation statistics
    print(f"Validation stats - Predictions: min={all_predictions.min():.4f}, max={all_predictions.max():.4f}, mean={all_predictions.mean():.4f}")
    print(f"Validation stats - Targets: min={all_targets.min():.4f}, max={all_targets.max():.4f}, mean={all_targets.mean():.4f}")
    print(f"Validation R² calculation: {r2_score:.4f}")
    if uncertainty_stats:
        print(f"Uncertainty stats: {uncertainty_stats}")
    
    return total_reward_mse / total_samples, total_reward_mae / total_samples, r2_score, uncertainty_stats

def train_reward_predictor():
    """
    Train the SOTA reward predictor with advanced features:
    - Multi-scale attention
    - Uncertainty estimation
    - Contrastive learning
    - Curriculum learning
    - Advanced regularization
    """
    config = load_config()
    
    # Initialize wandb
    wandb_config = config.copy()
    wandb_available = True
    try:
        wandb.init(project="sota-reward-predictor", config=wandb_config, settings=wandb.Settings(init_timeout=180))
        print("Wandb initialized successfully!")
    except Exception as e:
        print(f"Wandb initialization failed: {e}")
        print("Continuing without wandb logging...")
        wandb_available = False
    
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
    
    # SOTA stabilization parameters
    gradient_clip_norm = config.get('gradient_clip_norm', 1.0)
    patience = config.get('early_stopping_patience', 15)
    warmup_epochs = config.get('warmup_epochs', 5)

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

    # Create SOTA reward predictor
    reward_predictor = RewardPredictor(
        latent_dim=latent_dim,
        hidden_dim=config['reward_predictor'].get('hidden_dim', 256),
        transformer_depth=config['reward_predictor'].get('transformer_depth', 4),
        transformer_heads=config['reward_predictor'].get('transformer_heads', 8),
        transformer_dim_head=config['reward_predictor'].get('transformer_dim_head', 64),
        transformer_mlp_dim=config['reward_predictor'].get('transformer_mlp_dim', 512),
        dropout=config['reward_predictor'].get('transformer_dropout', 0.1),
        proj_dim=config['reward_predictor'].get('proj_dim', None),
        use_uncertainty=config['reward_predictor'].get('use_uncertainty', True),
        use_contrastive=config['reward_predictor'].get('use_contrastive', True),
        use_multi_scale=config['reward_predictor'].get('use_multi_scale', True)
    ).to(device)
    print(f"[SOTA RewardPredictor] Number of parameters: {sum(p.numel() for p in reward_predictor.parameters())}")

    # SOTA loss function
    reward_criterion = RewardPredictorLoss(
        mse_weight=config.get('mse_weight', 1.0),
        uncertainty_weight=config.get('uncertainty_weight', 0.1),
        contrastive_weight=config.get('contrastive_weight', 0.05)
    )
    
    # Optimizer with different learning rates for different components
    if use_pretrained_encoder and freeze_pretrained_encoder:
        optimizer = optim.AdamW(
            list(reward_predictor.parameters()), 
            lr=learning_rate,
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )
    else:
        # Different learning rates for encoder and predictor
        encoder_params = list(state_encoder.parameters())
        predictor_params = list(reward_predictor.parameters())
        
        optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': learning_rate * 0.1},  # Lower LR for encoder
            {'params': predictor_params, 'lr': learning_rate}
        ], weight_decay=1e-4, betas=(0.9, 0.999))

    # Advanced learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=[learning_rate * 0.1, learning_rate],
        total_steps=total_steps,
        pct_start=0.1,  # Warmup for 10% of training
        anneal_strategy='cos'
    )

    # Training loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    save_path = os.path.join('weights', 'best_model_reward_predictor.pth')
    os.makedirs('weights', exist_ok=True)
    
    print(f"Starting SOTA training with {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    print(f"SOTA features:")
    print(f"  - Multi-scale attention: {reward_predictor.use_multi_scale}")
    print(f"  - Uncertainty estimation: {reward_predictor.use_uncertainty}")
    print(f"  - Contrastive learning: {reward_predictor.use_contrastive}")
    print(f"  - Gradient clipping: {gradient_clip_norm}")
    print(f"  - Early stopping patience: {patience}")
    print(f"  - OneCycleLR scheduler with warmup")
    
    # Track global step for proper logging
    global_step = 0
    
    for epoch in range(num_epochs):
        state_encoder.train()
        reward_predictor.train()
        total_reward_mse = 0
        total_reward_mae = 0
        total_reward_r2 = 0
        total_samples = 0
        
        # Loss components tracking
        total_mse_loss = 0
        total_uncertainty_loss = 0
        total_contrastive_loss = 0
        
        # For R² calculation during training
        train_predictions = []
        train_targets = []
        
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

            # Predict reward with SOTA model
            pred_reward, uncertainty, contrastive_outputs = reward_predictor(latent_t, latent_tp1, latent_target)
            
            # Compute SOTA loss
            total_loss, loss_components = reward_criterion(pred_reward, reward, uncertainty, contrastive_outputs)
            
            # Compute MSE and MAE for metrics
            reward_mse = F.mse_loss(pred_reward.squeeze(-1), reward)
            reward_mae = F.l1_loss(pred_reward.squeeze(-1), reward)

            # Backward pass with gradient clipping
            optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping for all parameters
            if use_pretrained_encoder and freeze_pretrained_encoder:
                torch.nn.utils.clip_grad_norm_(reward_predictor.parameters(), max_norm=gradient_clip_norm)
            else:
                torch.nn.utils.clip_grad_norm_(
                    list(state_encoder.parameters()) + list(reward_predictor.parameters()),
                    max_norm=gradient_clip_norm
                )
            
            optimizer.step()
            scheduler.step()
            global_step += 1

            # EMA update for target encoder
            with torch.no_grad():
                for target_param, source_param in zip(target_encoder.parameters(), state_encoder.parameters()):
                    target_param.data.mul_(0.995).add_(source_param.data, alpha=1 - 0.995)

            # Collect predictions and targets for R² calculation
            train_predictions.append(pred_reward.squeeze(-1).detach())
            train_targets.append(reward)

            # Calculate batch R²
            batch_r2 = calculate_r2_score(reward, pred_reward.squeeze(-1))
            
            # Accumulate metrics
            total_reward_mse += reward_mse.item() * state.size(0)
            total_reward_mae += reward_mae.item() * state.size(0)
            total_reward_r2 += batch_r2 * state.size(0)
            total_samples += state.size(0)
            
            # Accumulate loss components
            total_mse_loss += loss_components['mse_loss'] * state.size(0)
            total_uncertainty_loss += loss_components['uncertainty_loss'] * state.size(0)
            total_contrastive_loss += loss_components['contrastive_loss'] * state.size(0)

            # Log batch metrics every log_interval steps
            if global_step % log_interval == 0 and wandb_available:
                wandb.log({
                    "step": global_step,
                    "epoch": epoch + 1,
                    "batch_reward_mse": reward_mse.item(),
                    "batch_reward_mae": reward_mae.item(),
                    "batch_reward_r2": batch_r2,
                    "batch_total_loss": total_loss.item(),
                    "batch_mse_loss": loss_components['mse_loss'],
                    "batch_uncertainty_loss": loss_components['uncertainty_loss'],
                    "batch_contrastive_loss": loss_components['contrastive_loss'],
                    "learning_rate": optimizer.param_groups[0]['lr'],
                })
                print(f"Batch {global_step} - MSE: {reward_mse.item():.4f}, MAE: {reward_mae.item():.4f}, R²: {batch_r2:.4f}, Total Loss: {total_loss.item():.4f}")

        # Compute average training metrics
        avg_reward_mse = total_reward_mse / total_samples
        avg_reward_mae = total_reward_mae / total_samples
        avg_reward_r2 = total_reward_r2 / total_samples
        
        # Average loss components
        avg_mse_loss = total_mse_loss / total_samples
        avg_uncertainty_loss = total_uncertainty_loss / total_samples
        avg_contrastive_loss = total_contrastive_loss / total_samples
        
        # Calculate R² for training data (using all predictions vs targets)
        train_predictions = torch.cat(train_predictions, dim=0)
        train_targets = torch.cat(train_targets, dim=0)
        train_r2 = calculate_r2_score(train_targets, train_predictions)
        
        # Debug: Print some statistics
        print(f"Training stats - Predictions: min={train_predictions.min():.4f}, max={train_predictions.max():.4f}, mean={train_predictions.mean():.4f}")
        print(f"Training stats - Targets: min={train_targets.min():.4f}, max={train_targets.max():.4f}, mean={train_targets.mean():.4f}")
        print(f"Training R² calculation: {train_r2:.4f}")

        # Evaluate on validation set
        val_reward_mse, val_reward_mae, val_r2, val_uncertainty_stats = evaluate_reward_predictor(
            reward_predictor, state_encoder, target_encoder, val_loader, device, reward_criterion
        )

        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train - MSE: {avg_reward_mse:.4f}, MAE: {avg_reward_mae:.4f}, Avg Batch R²: {avg_reward_r2:.4f}, Overall R²: {train_r2:.4f}")
        print(f"  Train Loss Components - MSE: {avg_mse_loss:.4f}, Uncertainty: {avg_uncertainty_loss:.4f}, Contrastive: {avg_contrastive_loss:.4f}")
        print(f"  Val   - MSE: {val_reward_mse:.4f}, MAE: {val_reward_mae:.4f}, R²: {val_r2:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Log validation metrics to wandb
        if wandb_available:
            log_dict = {
                "step": global_step,
                "epoch": epoch + 1,
                "train_reward_mse": avg_reward_mse,
                "train_reward_mae": avg_reward_mae,
                "train_reward_r2_avg_batch": avg_reward_r2,
                "train_reward_r2_overall": train_r2,
                "train_mse_loss": avg_mse_loss,
                "train_uncertainty_loss": avg_uncertainty_loss,
                "train_contrastive_loss": avg_contrastive_loss,
                "val_reward_mse": val_reward_mse,
                "val_reward_mae": val_reward_mae,
                "val_reward_r2": val_r2,
                "learning_rate": optimizer.param_groups[0]['lr']
            }
            
            # Add uncertainty stats if available
            if val_uncertainty_stats:
                log_dict.update({
                    f"val_{k}": v for k, v in val_uncertainty_stats.items()
                })
            
            print(f"Logging to wandb: {log_dict}")
            wandb.log(log_dict)
            print(f"Successfully logged metrics to wandb - Avg Batch R²: {avg_reward_r2:.4f}, Overall R²: {train_r2:.4f}, Val R²: {val_r2:.4f}")
        else:
            print(f"Wandb not available - Avg Batch R²: {avg_reward_r2:.4f}, Overall R²: {train_r2:.4f}, Val R²: {val_r2:.4f}")

        # Save best model
        if val_reward_mse < best_val_loss:
            best_val_loss = val_reward_mse
            epochs_no_improve = 0
            torch.save({
                'state_encoder': state_encoder.state_dict(),
                'reward_predictor': reward_predictor.state_dict(),
                'target_encoder': target_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'config': config
            }, save_path)
            print(f"New best model saved to {save_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
            
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation MSE for {patience} epochs.")
            break

    if wandb_available:
        wandb.finish()
    print("SOTA training completed!")

if __name__ == "__main__":
    train_reward_predictor()
