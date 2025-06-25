import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
from src.models.state_encoder import StateEncoder
from src.models.color_predictor import ColorPredictor
from src.models.mask_encoder import MaskEncoder
from src.models.selection_mask_predictor import SelectionMaskPredictor
from src.models.next_state_predictor import NextStatePredictor
from src.losses.vicreg import VICRegLoss
from src.data import ReplayBufferDataset

# --- Config Loader ---
def load_config(config_path="unified_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# --- One-hot encoding utility ---
def one_hot(indices, num_classes):
    return torch.nn.functional.one_hot(indices, num_classes=num_classes).float()

# --- Validation Metrics ---
def evaluate_all_modules(color_predictor, selection_predictor, next_state_predictor, state_encoder, mask_encoder, 
                        dataloader, device, color_criterion, num_color_selection_fns, num_selection_fns, num_transform_actions, 
                        use_vicreg_selection, use_vicreg_next_state, vicreg_loss_fn_selection=None, vicreg_loss_fn_next_state=None):
    color_predictor.eval()
    selection_predictor.eval()
    next_state_predictor.eval()
    state_encoder.eval()
    mask_encoder.eval()
    total_color_loss = 0
    total_selection_loss = 0
    total_next_state_loss = 0
    color_correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            state = batch['state'].to(device)
            next_state = batch['next_state'].to(device)
            action_colour = batch['action_colour'].to(device)
            action_selection = batch['action_selection'].to(device)
            action_transform = batch['action_transform'].to(device)
            target_colour = batch['colour'].to(device)
            selection_mask = batch['selection_mask'].to(device)
            
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
            
            # Selection mask prediction
            selection_input = torch.cat([latent, action_selection_onehot], dim=1)
            pred_latent_mask = selection_predictor(selection_input)
            target_latent_mask = mask_encoder(selection_mask.float())
            
            if use_vicreg_selection and vicreg_loss_fn_selection is not None:
                selection_loss, _, _, _ = vicreg_loss_fn_selection(pred_latent_mask, target_latent_mask)
            else:
                selection_loss = nn.MSELoss()(pred_latent_mask, target_latent_mask)
            
            # Next state prediction
            pred_next_latent = next_state_predictor(latent, action_transform_onehot, pred_latent_mask)
            
            if use_vicreg_next_state and vicreg_loss_fn_next_state is not None:
                next_state_loss, _, _, _ = vicreg_loss_fn_next_state(pred_next_latent, latent_next)
            else:
                next_state_loss = nn.MSELoss()(pred_next_latent, latent_next)
            
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
    All model choices and hyperparameters are loaded from unified_config.yaml.
    """
    config = load_config()
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
    selection_mask_predictor_hidden_dim = selection_cfg['selection_mask_predictor_hidden_dim']
    use_transformer_selection = selection_cfg['use_transformer']
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

    # Initialize all models
    state_encoder = StateEncoder(encoder_type, latent_dim=latent_dim, **encoder_params).to(device)
    color_predictor = ColorPredictor(latent_dim + num_color_selection_fns, num_arc_colors, color_predictor_hidden_dim).to(device)
    mask_encoder = MaskEncoder(mask_encoder_type, latent_dim=latent_mask_dim, **mask_encoder_params).to(device)
    selection_mask_predictor = SelectionMaskPredictor(
        input_dim=latent_dim + num_selection_fns,
        latent_mask_dim=latent_mask_dim,
        hidden_dim=selection_mask_predictor_hidden_dim,
        use_transformer=use_transformer_selection,
        transformer_depth=transformer_depth_selection,
        transformer_heads=transformer_heads_selection,
        transformer_dim_head=transformer_dim_head_selection,
        transformer_mlp_dim=transformer_mlp_dim_selection,
        dropout=transformer_dropout_selection
    ).to(device)
    next_state_predictor = NextStatePredictor(
        latent_dim=latent_dim,
        num_transform_actions=num_transform_actions,
        latent_mask_dim=latent_mask_dim_next_state,
        transformer_depth=transformer_depth_next_state,
        transformer_heads=transformer_heads_next_state,
        transformer_dim_head=transformer_dim_head_next_state,
        transformer_mlp_dim=transformer_mlp_dim_next_state,
        dropout=transformer_dropout_next_state
    ).to(device)

    # Loss functions
    color_criterion = nn.CrossEntropyLoss()
    selection_criterion = nn.MSELoss()
    next_state_criterion = nn.MSELoss()
    vicreg_loss_fn_selection = VICRegLoss(sim_coeff=vicreg_sim_coeff_selection, std_coeff=vicreg_std_coeff_selection, cov_coeff=vicreg_cov_coeff_selection)
    vicreg_loss_fn_next_state = VICRegLoss(sim_coeff=vicreg_sim_coeff_next_state, std_coeff=vicreg_std_coeff_next_state, cov_coeff=vicreg_cov_coeff_next_state)
    
    # Optimize all modules together
    optimizer = optim.Adam(
        list(state_encoder.parameters()) + 
        list(color_predictor.parameters()) + 
        list(mask_encoder.parameters()) + 
        list(selection_mask_predictor.parameters()) + 
        list(next_state_predictor.parameters()), 
        lr=learning_rate
    )

    best_val_loss = float('inf')
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
            
            # Selection mask prediction
            selection_input = torch.cat([latent, action_selection_onehot], dim=1)
            pred_latent_mask = selection_mask_predictor(selection_input)
            target_latent_mask = mask_encoder(selection_mask.float())
            
            if use_vicreg_selection:
                selection_loss, _, _, _ = vicreg_loss_fn_selection(pred_latent_mask, target_latent_mask)
            else:
                selection_loss = selection_criterion(pred_latent_mask, target_latent_mask)
            
            # Next state prediction
            pred_next_latent = next_state_predictor(latent, action_transform_onehot, pred_latent_mask)
            
            if use_vicreg_next_state:
                next_state_loss, _, _, _ = vicreg_loss_fn_next_state(pred_next_latent, latent_next)
            else:
                next_state_loss = next_state_criterion(pred_next_latent, latent_next)
            
            # Combined loss
            total_loss = color_loss + selection_loss + next_state_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_color_loss += color_loss.item() * state.size(0)
            total_selection_loss += selection_loss.item() * state.size(0)
            total_next_state_loss += next_state_loss.item() * state.size(0)
            
            if (i + 1) % log_interval == 0:
                print(f"Epoch {epoch+1} Batch {i+1}/{len(train_loader)} - Color: {color_loss.item():.4f} | Selection: {selection_loss.item():.4f} | Next State: {next_state_loss.item():.4f}")

        avg_color_loss = total_color_loss / len(train_loader.dataset)
        avg_selection_loss = total_selection_loss / len(train_loader.dataset)
        avg_next_state_loss = total_next_state_loss / len(train_loader.dataset)
        
        val_color_loss, val_selection_loss, val_next_state_loss, val_color_acc = evaluate_all_modules(
            color_predictor, selection_mask_predictor, next_state_predictor, state_encoder, mask_encoder,
            val_loader, device, color_criterion, num_color_selection_fns, num_selection_fns, num_transform_actions,
            use_vicreg_selection, use_vicreg_next_state, vicreg_loss_fn_selection, vicreg_loss_fn_next_state
        )
        
        total_val_loss = val_color_loss + val_selection_loss + val_next_state_loss
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train - Color: {avg_color_loss:.4f} | Selection: {avg_selection_loss:.4f} | Next State: {avg_next_state_loss:.4f}")
        print(f"Val - Color: {val_color_loss:.4f} | Selection: {val_selection_loss:.4f} | Next State: {val_next_state_loss:.4f} | Color Acc: {val_color_acc:.4f}")
        
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            torch.save({
                'state_encoder': state_encoder.state_dict(),
                'color_predictor': color_predictor.state_dict(),
                'mask_encoder': mask_encoder.state_dict(),
                'selection_mask_predictor': selection_mask_predictor.state_dict(),
                'next_state_predictor': next_state_predictor.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }, 'best_model_end_to_end.pth')
            print(f"Saved best model with validation loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train_next_state_predictor() 