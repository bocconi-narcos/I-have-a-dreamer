import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
import wandb
from src.models.state_encoder import StateEncoder
from src.models.predictors.color_predictor import ColorPredictor
from src.models.mask_encoder_new import MaskEncoder
from src.models.predictors.selection_mask_predictor import SelectionMaskPredictor
from src.models.predictors.next_state_predictor import NextStatePredictor
from src.losses.vicreg import VICRegLoss  # Not used, but kept for reference
from src.data import ReplayBufferDataset
from src.models.action_embed import ActionEmbedder

# --- EMA Utility ---
def update_ema(target_model, source_model, decay=0.995):
    with torch.no_grad():
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.mul_(decay).add_(source_param.data, alpha=1 - decay)

# --- Config Loader ---
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# --- One-hot encoding utility ---
def one_hot(indices, num_classes):
    return torch.nn.functional.one_hot(indices, num_classes=num_classes).float()

# --- Validation Metrics ---
def evaluate_all_modules(color_predictor, selection_predictor, next_state_predictor, state_encoder, target_encoder, mask_encoder, 
                        colour_selection_embedder, selection_embedder,
                        dataloader, device, color_criterion, num_color_selection_fns, num_selection_fns, num_transform_actions):
    color_predictor.eval()
    selection_predictor.eval()
    next_state_predictor.eval()
    state_encoder.eval()
    target_encoder.eval()
    mask_encoder.eval()
    total_color_loss = 0
    total_selection_loss = 0
    total_next_state_loss = 0
    color_correct = 0
    total = 0
    selection_criterion = nn.MSELoss()
    next_state_criterion = nn.MSELoss()
    with torch.no_grad():
        for batch in dataloader:
            state = batch['state'].to(device)
            next_state = batch['next_state'].to(device)
            action_colour = batch['action_colour'].to(device)
            action_selection = batch['action_selection'].to(device)
            action_transform = batch['action_transform'].to(device)
            target_colour = batch['colour'].to(device)
            selection_mask = batch['selection_mask'].to(device)
            shape_h = batch.get('shape_h', None)
            shape_w = batch.get('shape_w', None)
            num_colors_grid = batch.get('num_colors_grid', None)
            most_present_color = batch.get('most_present_color', None)
            least_present_color = batch.get('least_present_color', None)

            if state.dim() == 3:
                state = state.unsqueeze(1)
                next_state = next_state.unsqueeze(1)
                selection_mask = selection_mask.unsqueeze(1)

            # Encode state and next_state
            if shape_h is not None:
                latent = state_encoder(state.to(torch.long), shape_h=shape_h.to(device), shape_w=shape_w.to(device), num_unique_colors=num_colors_grid.to(device), most_common_color=most_present_color.to(device), least_common_color=least_present_color.to(device))
                latent_next = target_encoder(next_state.to(torch.long), shape_h=shape_h.to(device), shape_w=shape_w.to(device), num_unique_colors=num_colors_grid.to(device), most_common_color=most_present_color.to(device), least_common_color=least_present_color.to(device))
            else:
                latent = state_encoder(state.to(torch.long))
                latent_next = target_encoder(next_state.to(torch.long))

            action_colour_onehot = one_hot(action_colour, num_color_selection_fns)
            action_selection_onehot = one_hot(action_selection, num_selection_fns)
            action_transform_onehot = one_hot(action_transform, num_transform_actions)

            # Color prediction - using embedded actions
            action_color_embedding = colour_selection_embedder(action_colour_onehot)
            color_logits = color_predictor(latent, action_color_embedding)
            color_loss = color_criterion(color_logits, target_colour)

            # Selection mask prediction - now using embedded selection actions
            action_selection_embedding = selection_embedder(action_selection_onehot)
            pred_latent_mask = selection_predictor(latent, action_selection_embedding, color_logits.softmax(dim=1))
            target_latent_mask = mask_encoder(selection_mask.to(torch.long))
            selection_loss = selection_criterion(pred_latent_mask, target_latent_mask)

            # Next state prediction - pass pred_latent_mask as final latent selection
            pred_next_latent = next_state_predictor(latent, action_transform_onehot, pred_latent_mask)
            next_state_loss = next_state_criterion(pred_next_latent, latent_next)

            total_color_loss += color_loss.item() * state.size(0)
            total_selection_loss += selection_loss.item() * state.size(0)
            total_next_state_loss += next_state_loss.item() * state.size(0)
            color_preds = torch.argmax(color_logits, dim=1)
            color_correct += (color_preds == target_colour).sum().item()
            total += state.size(0)

    avg_color_loss = total_color_loss / total
    avg_selection_loss = total_selection_loss / total
    avg_next_state_loss = total_next_state_loss / total
    color_accuracy = color_correct / total
    return avg_color_loss, avg_selection_loss, avg_next_state_loss, color_accuracy

# --- Main Training Loop ---
def train_next_state_predictor():
    """
    Main training loop for end-to-end training of all three modules: color predictor, selection mask predictor, and next state predictor.
    The buffer is expected to be a list of dicts with the required keys. The training loop:
        1. Extracts all action components and one-hot encodes them.
        2. Passes state through the configurable state encoder.
        3. For color: concatenates state embedding and color action encoding, passes through color predictor MLP.
        4. For selection: concatenates state embedding and selection action encoding, passes through selection mask predictor.
        5. For selection: passes ground truth selection_mask through mask encoder to get target latent mask.
        6. For next state: concatenates state embedding, transform action encoding, and predicted latent mask, passes through next state predictor.
        7. Computes losses for all three modules and backpropagates through all networks.
    All model choices and hyperparameters are loaded from config.yaml.
    """
    config = load_config()
    buffer_path = config['buffer_path']
    encoder_type = config['encoder_type']
    latent_dim = config['latent_dim']
    encoder_params = config['encoder_params']
    num_color_selection_fns = config['action_embedders']['action_color_embedder']['num_actions']
    num_selection_fns = config['action_embedders']['action_selection_embedder']['num_actions']
    num_transform_actions = config['action_embedders']['action_transform_embedder']['num_actions']
    num_arc_colors = config['num_arc_colors']
    color_predictor_hidden_dim = config['color_predictor']['hidden_dim']

    # Action embedder configurations
    color_selection_dim = config['action_embedders']['action_color_embedder']['embed_dim']
    selection_dim = config['action_embedders']['action_selection_embedder']['embed_dim']
    
    # Selection mask config
    selection_cfg = config['selection_mask']
    mask_encoder_params = selection_cfg['mask_encoder_params']
    latent_mask_dim = selection_cfg['latent_mask_dim']
    mask_predictor_params = selection_cfg['mask_predictor_params']
    transformer_depth_selection = mask_predictor_params['transformer_depth']
    transformer_heads_selection = mask_predictor_params['transformer_heads']
    transformer_dim_head_selection = mask_predictor_params['transformer_dim_head']
    transformer_mlp_dim_selection = mask_predictor_params['transformer_mlp_dim']
    transformer_dropout_selection = mask_predictor_params['transformer_dropout']
    
    # Next state config
    next_state_cfg = config['next_state']
    latent_mask_dim_next_state = next_state_cfg['latent_mask_dim']
    transformer_depth_next_state = next_state_cfg['transformer_depth']
    transformer_heads_next_state = next_state_cfg['transformer_heads']
    transformer_dim_head_next_state = next_state_cfg['transformer_dim_head']
    transformer_mlp_dim_next_state = next_state_cfg['transformer_mlp_dim']
    transformer_dropout_next_state = next_state_cfg['transformer_dropout']
    
    batch_size = config['batch_size']
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    num_workers = config['num_workers']
    log_interval = config['log_interval']

    # State shape (channels, H, W) or (H, W)
    image_size = encoder_params.get('image_size', [10, 10])
    input_channels = encoder_params.get('input_channels', 1)
    if isinstance(image_size, int):
        state_shape = (input_channels, image_size, image_size)
    else:
        state_shape = (input_channels, image_size[0], image_size[1])

    dataset = ReplayBufferDataset(
        buffer_path=buffer_path,
        num_color_selection_fns=num_color_selection_fns,
        num_selection_fns=num_selection_fns,
        num_transform_actions=num_transform_actions,
        num_arc_colors=num_arc_colors,
        state_shape=state_shape,
        mode='end_to_end'
    )
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Device selection: prefer MPS (Apple Silicon GPU), then CUDA, then CPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device('mps')
        print('Using device: MPS (Apple Silicon GPU)')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using device: CUDA')
    else:
        device = torch.device('cpu')
        print('Using device: CPU')

    # --- WANDB LOGIN ---
    # Initialize wandb
    wandb_config = config.copy()
    wandb.init(project="next-state-predictor", config=wandb_config)  # type: ignore

    # Initialize all models
    state_encoder = StateEncoder(
        image_size=image_size,
        input_channels=input_channels,
        latent_dim=latent_dim,
        encoder_params=encoder_params
    ).to(device)
    # Target encoder (EMA)
    target_encoder = copy.deepcopy(state_encoder)
    target_encoder.eval()
    for p in target_encoder.parameters():
        p.requires_grad = False

    # Only keep valid keys for MaskEncoder
    valid_keys = {'image_size', 'vocab_size', 'emb_dim', 'depth', 'heads', 'mlp_dim', 'dropout', 'emb_dropout', 'padding_value'}
    filtered_mask_encoder_params = {k: v for k, v in mask_encoder_params.items() if k in valid_keys}
    mask_encoder = MaskEncoder(**filtered_mask_encoder_params).to(device)
    
    color_predictor = ColorPredictor(latent_dim, num_arc_colors, color_predictor_hidden_dim, action_embedding_dim=color_selection_dim).to(device)
    print(f"[ColorPredictor] Number of parameters: {sum(p.numel() for p in color_predictor.parameters())}")
    
    # Create action embedders
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
    
    # Updated SelectionMaskPredictor initialization - now takes embedded selection actions
    selection_mask_predictor = SelectionMaskPredictor(
        state_dim=latent_dim,
        selection_action_embed_dim=selection_dim,  # Use embedded dimension instead of one-hot
        color_pred_dim=num_arc_colors,  # Assuming color prediction dimension
        latent_mask_dim=latent_mask_dim,
        transformer_depth=transformer_depth_selection,
        transformer_heads=transformer_heads_selection,
        transformer_dim_head=transformer_dim_head_selection,
        transformer_mlp_dim=transformer_mlp_dim_selection,
        dropout=transformer_dropout_selection
    ).to(device)
    print(f"[SelectionMaskPredictor] Number of parameters: {sum(p.numel() for p in selection_mask_predictor.parameters())}")
    
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
    print(f"[NextStatePredictor] Number of parameters: {sum(p.numel() for p in next_state_predictor.parameters())}")

    # Loss functions
    color_criterion = nn.CrossEntropyLoss()
    selection_criterion = nn.MSELoss()
    next_state_criterion = nn.MSELoss()
    
    # Optimize all modules together - now including both embedders
    optimizer = optim.AdamW(
        list(state_encoder.parameters()) + 
        list(color_predictor.parameters()) + 
        list(mask_encoder.parameters()) + 
        list(selection_mask_predictor.parameters()) + 
        list(next_state_predictor.parameters()) +
        list(colour_selection_embedder.parameters()) +
        list(selection_embedder.parameters()), 
        lr=learning_rate
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10
    save_path = 'best_model_next_state_predictor.pth'
    for epoch in range(num_epochs):
        state_encoder.train()
        color_predictor.train()
        mask_encoder.train()
        selection_mask_predictor.train()
        next_state_predictor.train()
        total_color_loss = 0
        total_selection_loss = 0
        total_next_state_loss = 0
        for i, batch in enumerate(train_loader):
            state = batch['state'].to(device)
            next_state = batch['next_state'].to(device)
            action_colour = batch['action_colour'].to(device)
            action_selection = batch['action_selection'].to(device)
            action_transform = batch['action_transform'].to(device)
            target_colour = batch['colour'].to(device)
            selection_mask = batch['selection_mask'].to(device)
            shape_h = batch.get('shape_h', None)
            shape_w = batch.get('shape_w', None)
            num_colors_grid = batch.get('num_colors_grid', None)
            most_present_color = batch.get('most_present_color', None)
            least_present_color = batch.get('least_present_color', None)

            if state.dim() == 3:
                state = state.unsqueeze(1)
                next_state = next_state.unsqueeze(1)
                selection_mask = selection_mask.unsqueeze(1)

            # Encode state and next_state
            if shape_h is not None:
                latent = state_encoder(state.to(torch.long), shape_h=shape_h.to(device), shape_w=shape_w.to(device), num_unique_colors=num_colors_grid.to(device), most_common_color=most_present_color.to(device), least_common_color=least_present_color.to(device))
                latent_next = target_encoder(next_state.to(torch.long), shape_h=shape_h.to(device), shape_w=shape_w.to(device), num_unique_colors=num_colors_grid.to(device), most_common_color=most_present_color.to(device), least_common_color=least_present_color.to(device))
            else:
                latent = state_encoder(state.to(torch.long))
                latent_next = target_encoder(next_state.to(torch.long))

            action_colour_onehot = one_hot(action_colour, num_color_selection_fns)
            action_selection_onehot = one_hot(action_selection, num_selection_fns)
            action_transform_onehot = one_hot(action_transform, num_transform_actions)

            # Color prediction - using embedded actions
            action_color_embedding = colour_selection_embedder(action_colour_onehot)
            color_logits = color_predictor(latent, action_color_embedding)
            color_loss = color_criterion(color_logits, target_colour)

            # Selection mask prediction - now using embedded selection actions
            action_selection_embedding = selection_embedder(action_selection_onehot)
            pred_latent_mask = selection_mask_predictor(latent, action_selection_embedding, color_logits.softmax(dim=1))
            target_latent_mask = mask_encoder(selection_mask.to(torch.long))
            selection_loss = selection_criterion(pred_latent_mask, target_latent_mask)

            # Next state prediction - pass pred_latent_mask as final latent selection
            pred_next_latent = next_state_predictor(latent, action_transform_onehot, pred_latent_mask)
            next_state_loss = next_state_criterion(pred_next_latent, latent_next)

            # Combined loss
            total_loss = color_loss + selection_loss + next_state_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # EMA update for target encoder
            update_ema(target_encoder, state_encoder, decay=0.995)

            total_color_loss += color_loss.item() * state.size(0)
            total_selection_loss += selection_loss.item() * state.size(0)
            total_next_state_loss += next_state_loss.item() * state.size(0)

            # if (i + 1) % log_interval == 0:
            #     print(f"Epoch {epoch+1} Batch {i+1}/{len(train_loader)} - Color: {color_loss.item():.4f} | Selection: {selection_loss.item():.4f} | Next State: {next_state_loss.item():.4f}")
                # --- WANDB LOGGING FOR BATCH ---
                # wandb.log({  # type: ignore
                #     "batch_color_loss": color_loss.item(),
                #     "batch_selection_loss": selection_loss.item(),
                #     "batch_next_state_loss": next_state_loss.item(),
                #     "batch_total_loss": total_loss.item(),
                #     "epoch": epoch + 1,
                #     "batch": i + 1
                # })

        avg_color_loss = total_color_loss / len(train_loader.dataset)
        avg_selection_loss = total_selection_loss / len(train_loader.dataset)
        avg_next_state_loss = total_next_state_loss / len(train_loader.dataset)

        val_color_loss, val_selection_loss, val_next_state_loss, val_color_acc = evaluate_all_modules(
            color_predictor, selection_mask_predictor, next_state_predictor, state_encoder, target_encoder, mask_encoder,
            colour_selection_embedder, selection_embedder,
            val_loader, device, color_criterion, num_color_selection_fns, num_selection_fns, num_transform_actions
        )

        val_loss_sum = val_color_loss + val_selection_loss + val_next_state_loss
        print(f"Epoch {epoch+1}/{num_epochs} - Train Color Loss: {avg_color_loss:.4f} | Train Selection Loss: {avg_selection_loss:.4f} | Train Next State Loss: {avg_next_state_loss:.4f} | Val Color Loss: {val_color_loss:.4f} | Val Selection Loss: {val_selection_loss:.4f} | Val Next State Loss: {val_next_state_loss:.4f} | Val Color Acc: {val_color_acc:.4f}")
        # --- WANDB LOGGING FOR EPOCH ---
        wandb.log({  # type: ignore
            "epoch": epoch + 1,
            "train_color_loss": avg_color_loss,
            "train_selection_loss": avg_selection_loss,
            "train_next_state_loss": avg_next_state_loss,
            "val_color_loss": val_color_loss,
            "val_selection_loss": val_selection_loss,
            "val_next_state_loss": val_next_state_loss,
            "val_color_acc": val_color_acc,
            "val_total_loss": val_loss_sum
        })
        if val_loss_sum < best_val_loss:
            best_val_loss = val_loss_sum
            epochs_no_improve = 0
            torch.save({
                'state_encoder': state_encoder.state_dict(),
                'color_predictor': color_predictor.state_dict(),
                'mask_encoder': mask_encoder.state_dict(),
                'selection_mask_predictor': selection_mask_predictor.state_dict(),
                'next_state_predictor': next_state_predictor.state_dict(),
                'colour_selection_embedder': colour_selection_embedder.state_dict(),
                'selection_embedder': selection_embedder.state_dict(),
                'target_encoder': target_encoder.state_dict()
            }, save_path)
            print(f"New best model saved to {save_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss for {patience} epochs.")
            break

    wandb.finish()  # type: ignore

if __name__ == "__main__":
    train_next_state_predictor() 