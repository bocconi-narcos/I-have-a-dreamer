import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
from src.models.state_encoder import StateEncoder
from models.predictors.color_predictor import ColorPredictor
from src.models.mask_encoder import MaskEncoder
from models.predictors.selection_mask_predictor import SelectionMaskPredictor
from models.predictors.next_state_predictor import NextStatePredictor
from models.predictors.reward_predictor import RewardPredictor
from models.predictors.continuation_predictor import ContinuationPredictor
from src.losses.vicreg import VICRegLoss
from src.data import ReplayBufferDataset
import traceback

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def one_hot(indices, num_classes):
    return torch.nn.functional.one_hot(indices, num_classes=num_classes).float()

def train_full_model():
    try:
        print("Loading configuration...")
        config = load_config()
        print("Configuration loaded successfully.")
        buffer_path = config['buffer_path']
        encoder_type = config['encoder_type']
        latent_dim = config['latent_dim']
        encoder_params = config['encoder_params'][encoder_type]
        num_color_selection_fns = config['num_color_selection_fns']
        num_selection_fns = config['num_selection_fns']
        num_transform_actions = config['num_transform_actions']
        num_arc_colors = config['num_arc_colors']
        color_predictor_hidden_dim = config['color_predictor']['hidden_dim']

        # Selection mask config
        selection_cfg = config['selection_mask']
        mask_encoder_type = selection_cfg['mask_encoder_type']
        mask_encoder_params = selection_cfg['mask_encoder_params'][mask_encoder_type]
        latent_mask_dim = selection_cfg['latent_mask_dim']
        transformer_depth_selection = selection_cfg['transformer_depth']
        transformer_heads_selection = selection_cfg['transformer_heads']
        transformer_dim_head_selection = selection_cfg['transformer_dim_head']
        transformer_mlp_dim_selection = selection_cfg['transformer_mlp_dim']
        transformer_dropout_selection = selection_cfg['transformer_dropout']
        use_vicreg_selection = selection_cfg['use_vicreg']
        vicreg_sim_coeff_selection = selection_cfg['vicreg_sim_coeff']
        vicreg_std_coeff_selection = selection_cfg['vicreg_std_coeff']
        vicreg_cov_coeff_selection = selection_cfg['vicreg_cov_coeff']

        # Next state config
        next_state_cfg = config['next_state']
        latent_mask_dim_next_state = next_state_cfg['latent_mask_dim']
        transformer_depth_next_state = next_state_cfg['transformer_depth']
        transformer_heads_next_state = next_state_cfg['transformer_heads']
        transformer_dim_head_next_state = next_state_cfg['transformer_dim_head']
        transformer_mlp_dim_next_state = next_state_cfg['transformer_mlp_dim']
        transformer_dropout_next_state = next_state_cfg['transformer_dropout']
        use_vicreg_next_state = next_state_cfg['use_vicreg']
        vicreg_sim_coeff_next_state = next_state_cfg['vicreg_sim_coeff']
        vicreg_std_coeff_next_state = next_state_cfg['vicreg_std_coeff']
        vicreg_cov_coeff_next_state = next_state_cfg['vicreg_cov_coeff']

        # Reward predictor config
        reward_cfg = config['reward_predictor']
        reward_latent_dim = reward_cfg['latent_dim']
        reward_hidden_dim = reward_cfg['hidden_dim']
        reward_transformer_depth = reward_cfg['transformer_depth']
        reward_transformer_heads = reward_cfg['transformer_heads']
        reward_transformer_dim_head = reward_cfg['transformer_dim_head']
        reward_transformer_mlp_dim = reward_cfg['transformer_mlp_dim']
        reward_transformer_dropout = reward_cfg['transformer_dropout']
        reward_proj_dim = reward_cfg.get('proj_dim', None)

        # Continuation predictor config
        continuation_cfg = config['continuation_predictor']
        continuation_latent_dim = continuation_cfg['latent_dim']
        continuation_hidden_dim = continuation_cfg['hidden_dim']
        continuation_transformer_depth = continuation_cfg['transformer_depth']
        continuation_transformer_heads = continuation_cfg['transformer_heads']
        continuation_transformer_dim_head = continuation_cfg['transformer_dim_head']
        continuation_transformer_mlp_dim = continuation_cfg['transformer_mlp_dim']
        continuation_transformer_dropout = continuation_cfg['transformer_dropout']
        continuation_proj_dim = continuation_cfg.get('proj_dim', None)

        batch_size = config['batch_size']
        num_epochs = config['num_epochs']
        learning_rate = config['learning_rate']
        num_workers = config['num_workers']
        log_interval = config['log_interval']

        image_size = encoder_params.get('image_size', [10, 10])
        input_channels = encoder_params.get('input_channels', 1)
        if isinstance(image_size, int):
            state_shape = (input_channels, image_size, image_size)
        else:
            state_shape = (input_channels, image_size[0], image_size[1])

        print("Preparing dataset...")
        dataset = ReplayBufferDataset(
            buffer_path=buffer_path,
            num_color_selection_fns=num_color_selection_fns,
            num_selection_fns=num_selection_fns,
            num_transform_actions=num_transform_actions,
            num_arc_colors=num_arc_colors,
            state_shape=state_shape,
            mode='end_to_end'
        )
        print(f"Dataset size: {len(dataset)}")
        val_size = int(0.2 * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        print("Dataset split into train and validation sets.")

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

        print("Initializing models...")
        state_encoder = StateEncoder(encoder_type, latent_dim=latent_dim, **encoder_params).to(device)
        color_predictor = ColorPredictor(latent_dim + num_color_selection_fns, num_arc_colors, color_predictor_hidden_dim).to(device)
        mask_encoder = MaskEncoder(mask_encoder_type, latent_dim=latent_mask_dim, **mask_encoder_params).to(device)
        
        # Updated SelectionMaskPredictor initialization
        selection_mask_predictor = SelectionMaskPredictor(
            state_dim=latent_dim,
            selection_action_dim=num_selection_fns,
            color_pred_dim=num_arc_colors,  # Assuming color prediction dimension
            latent_mask_dim=latent_mask_dim,
            transformer_depth=transformer_depth_selection,
            transformer_heads=transformer_heads_selection,
            transformer_dim_head=transformer_dim_head_selection,
            transformer_mlp_dim=transformer_mlp_dim_selection,
            dropout=transformer_dropout_selection
        ).to(device)
        
        # Updated NextStatePredictor initialization
        next_state_predictor = NextStatePredictor(
            state_dim=latent_dim,
            num_transform_actions=num_transform_actions,
            latent_mask_dim=latent_mask_dim_next_state,
            latent_dim=latent_dim,
            transformer_depth=transformer_depth_next_state,
            transformer_heads=transformer_heads_next_state,
            transformer_dim_head=transformer_dim_head_next_state,
            transformer_mlp_dim=transformer_mlp_dim_next_state,
            dropout=transformer_dropout_next_state
        ).to(device)
        
        # Updated RewardPredictor initialization
        reward_predictor = RewardPredictor(
            latent_dim=reward_latent_dim,
            hidden_dim=reward_hidden_dim,
            transformer_depth=reward_transformer_depth,
            transformer_heads=reward_transformer_heads,
            transformer_dim_head=reward_transformer_dim_head,
            transformer_mlp_dim=reward_transformer_mlp_dim,
            dropout=reward_transformer_dropout,
            proj_dim=reward_proj_dim
        ).to(device)
        
        # Updated ContinuationPredictor initialization
        continuation_predictor = ContinuationPredictor(
            latent_dim=continuation_latent_dim,
            hidden_dim=continuation_hidden_dim,
            transformer_depth=continuation_transformer_depth,
            transformer_heads=continuation_transformer_heads,
            transformer_dim_head=continuation_transformer_dim_head,
            transformer_mlp_dim=continuation_transformer_mlp_dim,
            dropout=continuation_transformer_dropout,
            proj_dim=continuation_proj_dim
        ).to(device)
        print("All models initialized successfully.")

        # Loss functions
        color_criterion = nn.CrossEntropyLoss()
        selection_criterion = nn.MSELoss()
        next_state_criterion = nn.MSELoss()
        reward_criterion = nn.MSELoss()
        continuation_criterion = nn.BCELoss()
        vicreg_loss_fn_selection = VICRegLoss(sim_coeff=vicreg_sim_coeff_selection, std_coeff=vicreg_std_coeff_selection, cov_coeff=vicreg_cov_coeff_selection)
        vicreg_loss_fn_next_state = VICRegLoss(sim_coeff=vicreg_sim_coeff_next_state, std_coeff=vicreg_std_coeff_next_state, cov_coeff=vicreg_cov_coeff_next_state)

        optimizer = optim.AdamW(
            list(state_encoder.parameters()) +
            list(color_predictor.parameters()) +
            list(mask_encoder.parameters()) +
            list(selection_mask_predictor.parameters()) +
            list(next_state_predictor.parameters()) +
            list(reward_predictor.parameters()) +
            list(continuation_predictor.parameters()),
            lr=learning_rate
        )

        best_val_loss = float('inf')
        epochs_no_improve = 0
        patience = 10
        save_path = 'best_model_full_model.pth'
        print("Starting training loop...")
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            state_encoder.train()
            color_predictor.train()
            mask_encoder.train()
            selection_mask_predictor.train()
            next_state_predictor.train()
            reward_predictor.train()
            continuation_predictor.train()
            total_color_loss = 0
            total_selection_loss = 0
            total_next_state_loss = 0
            total_reward_loss = 0
            total_continuation_loss = 0
            for i, batch in enumerate(train_loader):
                try:
                    print(f"\nBatch {i+1}:")
                    print(f"  State shape: {batch['state'].shape}")
                    print(f"  Next state shape: {batch['next_state'].shape}")
                    print(f"  Reward shape: {batch['reward'].shape}, Done shape: {batch['done'].shape}")

                    state = batch['state'].to(device)
                    next_state = batch['next_state'].to(device)
                    action_colour = batch['action_colour'].to(device)
                    action_selection = batch['action_selection'].to(device)
                    action_transform = batch['action_transform'].to(device)
                    target_colour = batch['colour'].to(device)
                    selection_mask = batch['selection_mask'].to(device)
                    reward = batch['reward'].to(device)
                    done = batch['done'].to(device)

                    if state.dim() == 3:
                        state = state.unsqueeze(1)
                        next_state = next_state.unsqueeze(1)
                        selection_mask = selection_mask.unsqueeze(1)

                    latent = state_encoder(state.float())
                    latent_next = state_encoder(next_state.float())
                    action_colour_onehot = one_hot(action_colour, num_color_selection_fns)
                    action_selection_onehot = one_hot(action_selection, num_selection_fns)
                    action_transform_onehot = one_hot(action_transform, num_transform_actions)

                    # Color prediction
                    color_input = torch.cat([latent, action_colour_onehot], dim=1)
                    color_logits = color_predictor(color_input)
                    color_loss = color_criterion(color_logits, target_colour)

                    # Selection mask prediction - updated to use new signature
                    pred_latent_mask = selection_mask_predictor(latent, action_selection_onehot, color_logits.softmax(dim=1))
                    target_latent_mask = mask_encoder(selection_mask.float())
                    if use_vicreg_selection:
                        selection_loss, _, _, _ = vicreg_loss_fn_selection(pred_latent_mask, target_latent_mask)
                    else:
                        selection_loss = selection_criterion(pred_latent_mask, target_latent_mask)

                    # Next state prediction - updated to use new signature
                    pred_next_latent = next_state_predictor(latent, action_transform_onehot, pred_latent_mask)
                    if use_vicreg_next_state:
                        next_state_loss, _, _, _ = vicreg_loss_fn_next_state(pred_next_latent, latent_next)
                    else:
                        next_state_loss = next_state_criterion(pred_next_latent, latent_next)

                    # Reward prediction - updated to use new signature
                    pred_reward = reward_predictor(latent, pred_next_latent)
                    reward_loss = reward_criterion(pred_reward, reward)

                    # Continuation prediction - updated to use new signature
                    pred_continuation = continuation_predictor(latent, pred_next_latent)
                    continuation_loss = continuation_criterion(pred_continuation, 1.0 - done)  # 1-done: 1=continue, 0=done

                    print(f"  Losses: color={color_loss.item():.4f}, selection={selection_loss.item():.4f}, next_state={next_state_loss.item():.4f}, reward={reward_loss.item():.4f}, continuation={continuation_loss.item():.4f}")

                    total_color_loss += color_loss.item() * state.size(0)
                    total_selection_loss += selection_loss.item() * state.size(0)
                    total_next_state_loss += next_state_loss.item() * state.size(0)
                    total_reward_loss += reward_loss.item() * state.size(0)
                    total_continuation_loss += continuation_loss.item() * state.size(0)

                    total_loss = color_loss + selection_loss + next_state_loss + reward_loss + continuation_loss
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    print("  Backward pass and optimizer step successful.")

                    if i % log_interval == 0:
                        print(f"Epoch {epoch+1} Batch {i+1}: Color Loss={color_loss.item():.4f}, Selection Loss={selection_loss.item():.4f}, Next State Loss={next_state_loss.item():.4f}, Reward Loss={reward_loss.item():.4f}, Continuation Loss={continuation_loss.item():.4f}, Total Loss={total_loss.item():.4f}")

                except Exception as batch_err:
                    print(f"Error in batch {i+1} of epoch {epoch+1}: {batch_err}")
                    traceback.print_exc()
                    continue  # Skip to next batch

            print(f"Epoch {epoch+1} Summary: Color Loss={total_color_loss/train_size:.4f}, Selection Loss={total_selection_loss/train_size:.4f}, Next State Loss={total_next_state_loss/train_size:.4f}, Reward Loss={total_reward_loss/train_size:.4f}, Continuation Loss={total_continuation_loss/train_size:.4f}")

            # Validation after each epoch
            state_encoder.eval()
            color_predictor.eval()
            mask_encoder.eval()
            selection_mask_predictor.eval()
            next_state_predictor.eval()
            reward_predictor.eval()
            continuation_predictor.eval()
            val_color_loss = 0
            val_selection_loss = 0
            val_next_state_loss = 0
            val_reward_loss = 0
            val_continuation_loss = 0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    try:
                        state = batch['state'].to(device)
                        next_state = batch['next_state'].to(device)
                        action_colour = batch['action_colour'].to(device)
                        action_selection = batch['action_selection'].to(device)
                        action_transform = batch['action_transform'].to(device)
                        target_colour = batch['colour'].to(device)
                        selection_mask = batch['selection_mask'].to(device)
                        reward = batch['reward'].to(device)
                        done = batch['done'].to(device)
                        if state.dim() == 3:
                            state = state.unsqueeze(1)
                            next_state = next_state.unsqueeze(1)
                            selection_mask = selection_mask.unsqueeze(1)
                        latent = state_encoder(state.float())
                        latent_next = state_encoder(next_state.float())
                        action_colour_onehot = one_hot(action_colour, num_color_selection_fns)
                        action_selection_onehot = one_hot(action_selection, num_selection_fns)
                        action_transform_onehot = one_hot(action_transform, num_transform_actions)
                        color_input = torch.cat([latent, action_colour_onehot], dim=1)
                        color_logits = color_predictor(color_input)
                        color_loss = color_criterion(color_logits, target_colour)
                        
                        # Updated validation calls
                        pred_latent_mask = selection_mask_predictor(latent, action_selection_onehot, color_logits.softmax(dim=1))
                        target_latent_mask = mask_encoder(selection_mask.float())
                        if use_vicreg_selection:
                            selection_loss, _, _, _ = vicreg_loss_fn_selection(pred_latent_mask, target_latent_mask)
                        else:
                            selection_loss = selection_criterion(pred_latent_mask, target_latent_mask)
                        pred_next_latent = next_state_predictor(latent, action_transform_onehot, pred_latent_mask)
                        if use_vicreg_next_state:
                            next_state_loss, _, _, _ = vicreg_loss_fn_next_state(pred_next_latent, latent_next)
                        else:
                            next_state_loss = next_state_criterion(pred_next_latent, latent_next)
                        pred_reward = reward_predictor(latent, pred_next_latent)
                        reward_loss = reward_criterion(pred_reward, reward)
                        pred_continuation = continuation_predictor(latent, pred_next_latent)
                        continuation_loss = continuation_criterion(pred_continuation, 1.0 - done)
                        val_color_loss += color_loss.item() * state.size(0)
                        val_selection_loss += selection_loss.item() * state.size(0)
                        val_next_state_loss += next_state_loss.item() * state.size(0)
                        val_reward_loss += reward_loss.item() * state.size(0)
                        val_continuation_loss += continuation_loss.item() * state.size(0)
                        val_batches += state.size(0)
                    except Exception as batch_err:
                        print(f"Error in validation batch: {batch_err}")
                        continue
            val_color_loss /= val_batches
            val_selection_loss /= val_batches
            val_next_state_loss /= val_batches
            val_reward_loss /= val_batches
            val_continuation_loss /= val_batches
            val_loss_sum = val_color_loss + val_selection_loss + val_next_state_loss + val_reward_loss + val_continuation_loss
            print(f"Validation: Color Loss={val_color_loss:.4f}, Selection Loss={val_selection_loss:.4f}, Next State Loss={val_next_state_loss:.4f}, Reward Loss={val_reward_loss:.4f}, Continuation Loss={val_continuation_loss:.4f}, Total Val Loss={val_loss_sum:.4f}")
            if val_loss_sum < best_val_loss:
                best_val_loss = val_loss_sum
                epochs_no_improve = 0
                torch.save({
                    'state_encoder': state_encoder.state_dict(),
                    'color_predictor': color_predictor.state_dict(),
                    'mask_encoder': mask_encoder.state_dict(),
                    'selection_mask_predictor': selection_mask_predictor.state_dict(),
                    'next_state_predictor': next_state_predictor.state_dict(),
                    'reward_predictor': reward_predictor.state_dict(),
                    'continuation_predictor': continuation_predictor.state_dict()
                }, save_path)
                print(f"New best model saved to {save_path}")
            else:
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss for {patience} epochs.")
                break

    except Exception as e:
        print(f"Fatal error in training loop: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    train_full_model() 