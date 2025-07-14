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
from src.models.predictors.reward_predictor import RewardPredictor
from src.models.state_decoder import StateDecoder
from src.models.mask_decoder_new import MaskDecoder
from src.losses.vicreg import VICRegLoss  # Not used, but kept for reference
from src.data import ReplayBufferDataset
from src.models.action_embed import ActionEmbedder
from tqdm import tqdm
import torch.nn.functional as F  # type: ignore
import torch.nn.utils
import subprocess

# --- EMA Utility ---
def update_ema(target_model, source_model, decay=0.995):
    with torch.no_grad():
        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.mul_(decay).add_(source_param.data, alpha=1 - decay)

# --- Config Loader ---
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Add default values for ground truth and decoder switches
    config.setdefault('use_ground_truth', False)  # Use ground truth inputs instead of predicted
    config.setdefault('use_decoder_loss', False)   # Use decoder losses instead of latent space losses
    
    return config

# --- One-hot encoding utility ---
def one_hot(indices, num_classes):
    return torch.nn.functional.one_hot(indices, num_classes=num_classes).float()

# --- Autoencoder loss function ---
def autoencoder_loss(decoder_output, original_grid, shape_h, shape_w, most_common, least_common, unique_count):
    mask = (original_grid != -1)
    grid_logits = decoder_output['grid_logits']
    B, H, W, C = grid_logits.shape
    grid_loss = F.cross_entropy(
        grid_logits.reshape(B*H*W, C),
        (original_grid + 1).clamp(min=0).reshape(B*H*W),
        reduction='none'
    ).view(B, H, W)
    grid_loss = (grid_loss * mask).sum() / mask.sum()

    shape_h_loss = F.cross_entropy(decoder_output['shape_h_logits'], shape_h - 1)
    shape_w_loss = F.cross_entropy(decoder_output['shape_w_logits'], shape_w - 1)
    mc_loss = F.cross_entropy(decoder_output['most_common_logits'], most_common)
    lc_loss = F.cross_entropy(decoder_output['least_common_logits'], least_common)
    uc_loss = F.cross_entropy(decoder_output['unique_count_logits'], unique_count)

    total_loss = grid_loss + 0.25 * (shape_h_loss + shape_w_loss ) + 0.1* ( mc_loss + lc_loss + uc_loss)

    return total_loss, {
        'grid_loss': grid_loss.item(),
        'shape_loss': (shape_h_loss + shape_w_loss).item() / 2,
        'color_stats_loss': (mc_loss + lc_loss + uc_loss).item() / 3
    }

# --- Validation Metrics ---
def evaluate_all_modules(color_predictor, selection_predictor, next_state_predictor, state_encoder, target_encoder, mask_encoder, 
                        colour_selection_embedder, selection_embedder, dataloader, device, color_criterion, num_color_selection_fns, num_selection_fns, num_transform_actions,
                        use_vicreg_selection, vicreg_loss_fn_selection, selection_criterion, use_vicreg_next_state, vicreg_loss_fn_next_state, next_state_criterion,
                        state_decoder=None, mask_decoder=None, use_ground_truth=False, use_decoder_loss=False, num_arc_colors=None):
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
    # For per-class accuracy
    color_class_correct = None
    color_class_total = None
    # For next state predictor metrics
    total_next_state_cosine = 0
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
            color_preds = torch.argmax(color_logits, dim=1)
            color_correct += (color_preds == target_colour).sum().item()
            # Per-class accuracy
            if color_logits is not None and color_class_correct is None:
                color_class_correct = torch.zeros(color_logits.shape[1], device=device)
                color_class_total = torch.zeros(color_logits.shape[1], device=device)
            if color_logits is not None and color_class_correct is not None and color_class_total is not None:
                for c in range(color_logits.shape[1]):
                    color_class_correct[c] += ((color_preds == c) & (target_colour == c)).sum().item()
                    color_class_total[c] += (target_colour == c).sum().item()

            # Selection mask prediction - now using embedded selection actions
            action_selection_embedding = selection_embedder(action_selection_onehot)
            
            # Ground truth switch: use one-hot encoded color vs predicted color distribution
            if use_ground_truth:
                # Use ground truth one-hot encoded color (always use num_arc_colors)
                color_input = one_hot(target_colour, num_arc_colors)
            else:
                # Use predicted color distribution
                color_input = color_logits.softmax(dim=1)
            
            pred_latent_mask = selection_predictor(latent, action_selection_embedding, color_input)
            
            # Decoder switch: use decoder loss vs latent space loss
            if use_decoder_loss and mask_decoder is not None:
                # Decode predicted mask and compute loss against ground truth mask
                pred_mask_logits = mask_decoder(pred_latent_mask)
                # Compute cross-entropy loss on the decoded mask
                B, H, W, C = pred_mask_logits.shape
                mask_loss = F.cross_entropy(
                    pred_mask_logits.view(B*H*W, C),
                    selection_mask.view(B*H*W).long(),
                    reduction='mean'
                )
                selection_loss = mask_loss
            else:
                # Use latent space loss (original behavior)
                target_latent_mask = mask_encoder(selection_mask.to(torch.long))
                if use_vicreg_selection:
                    selection_loss, _, _, _ = vicreg_loss_fn_selection(pred_latent_mask, target_latent_mask)
                else:
                    selection_loss = selection_criterion(pred_latent_mask, target_latent_mask)

            # Next state prediction - handle ground truth vs predicted inputs
            # Ground truth switch: use encoded ground truth mask vs predicted mask
            if use_ground_truth:
                # Use ground truth encoded mask
                mask_input = mask_encoder(selection_mask.to(torch.long))
            else:
                # Use predicted mask
                mask_input = pred_latent_mask
            
            pred_next_latent = next_state_predictor(latent, action_transform_onehot, mask_input)
            
            # Decoder switch: use decoder loss vs latent space loss
            if use_decoder_loss and state_decoder is not None:
                # Decode predicted next state and compute loss against ground truth next state
                pred_next_state_logits = state_decoder(pred_next_latent)
                # Compute autoencoder-style loss
                next_state_loss, _ = autoencoder_loss(
                    pred_next_state_logits, 
                    next_state, 
                    shape_h.to(device), 
                    shape_w.to(device), 
                    most_present_color.to(device), 
                    least_present_color.to(device), 
                    num_colors_grid.to(device)
                )
            else:
                # Use latent space loss (original behavior)
                if use_vicreg_next_state:
                    next_state_loss, _, _, _ = vicreg_loss_fn_next_state(pred_next_latent, latent_next)
                else:
                    next_state_loss = next_state_criterion(pred_next_latent, latent_next)

            # Cosine similarity for next state prediction
            pred_next_latent_norm = F.normalize(pred_next_latent, p=2, dim=-1)
            latent_next_norm = F.normalize(latent_next, p=2, dim=-1)
            cosine_sim = (pred_next_latent_norm * latent_next_norm).sum(dim=-1).mean().item()
            total_next_state_cosine += cosine_sim * state.size(0)

            total_color_loss += color_loss.item() * state.size(0)
            total_selection_loss += selection_loss.item() * state.size(0)
            total_next_state_loss += next_state_loss.item() * state.size(0)
            total += state.size(0)

    avg_color_loss = total_color_loss / total
    avg_selection_loss = total_selection_loss / total
    avg_next_state_loss = total_next_state_loss / total
    color_accuracy = color_correct / total
    # Per-class accuracy
    if (
        color_class_correct is not None and color_class_total is not None
        and isinstance(color_class_correct, torch.Tensor)
        and isinstance(color_class_total, torch.Tensor)
        and color_class_correct.numel() > 0 and color_class_total.numel() > 0
    ):
        color_class_acc = (color_class_correct / (color_class_total + 1e-8)).tolist()
    else:
        color_class_acc = None
    avg_next_state_cosine = total_next_state_cosine / total
    return avg_color_loss, avg_selection_loss, avg_next_state_loss, color_accuracy, color_class_acc, avg_next_state_cosine

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
    fast_buffer_path = buffer_path + '.fast.pt'
    if not os.path.exists(fast_buffer_path):
        print(f"Fast buffer {fast_buffer_path} not found. Preprocessing...")
        subprocess.run(['python', 'scripts/preprocess_buffer.py', buffer_path, fast_buffer_path], check=True)
    else:
        print(f"Using fast buffer: {fast_buffer_path}")
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
    use_vicreg_selection = selection_cfg.get('use_vicreg', False)
    vicreg_sim_coeff_selection = selection_cfg.get('vicreg_sim_coeff', 1.0)
    vicreg_std_coeff_selection = selection_cfg.get('vicreg_std_coeff', 1.0)
    vicreg_cov_coeff_selection = selection_cfg.get('vicreg_cov_coeff', 1.0)
    
    # Next state config
    next_state_cfg = config['next_state']
    latent_mask_dim_next_state = next_state_cfg['latent_mask_dim']
    transformer_depth_next_state = next_state_cfg['transformer_depth']
    transformer_heads_next_state = next_state_cfg['transformer_heads']
    transformer_dim_head_next_state = next_state_cfg['transformer_dim_head']
    transformer_mlp_dim_next_state = next_state_cfg['transformer_mlp_dim']
    transformer_dropout_next_state = next_state_cfg['transformer_dropout']
    use_vicreg_next_state = next_state_cfg.get('use_vicreg', False)
    vicreg_sim_coeff_next_state = next_state_cfg.get('vicreg_sim_coeff', 1.0)
    vicreg_std_coeff_next_state = next_state_cfg.get('vicreg_std_coeff', 1.0)
    vicreg_cov_coeff_next_state = next_state_cfg.get('vicreg_cov_coeff', 1.0)
    
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
        buffer_path=fast_buffer_path,
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

    # --- Load pretrained encoder if specified ---
    use_pretrained_encoder = config.get('use_pretrained_encoder', False)
    pretrained_encoder_path = config.get('pretrained_encoder_path', 'best_model_autoencoder.pth')
    freeze_pretrained_encoder = config.get('freeze_pretrained_encoder', False)

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
                print("Encoder parameters frozen. Only decoder will be trained.")
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

    # Reward Predictor initialization
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

    # Load ground truth and decoder switches
    use_ground_truth = config.get('use_ground_truth', False)
    use_decoder_loss = config.get('use_decoder_loss', False)
    print(f"Training configuration:")
    print(f"  - Use ground truth inputs: {use_ground_truth}")
    print(f"  - Use decoder losses: {use_decoder_loss}")

    # Initialize decoders if using decoder losses
    state_decoder = None
    mask_decoder = None
    if use_decoder_loss:
        # State decoder for next state prediction
        state_decoder = StateDecoder(
            image_size=image_size,
            latent_dim=latent_dim,
            decoder_params=config.get('decoder_params', {})
        ).to(device)
        print(f"[StateDecoder] Number of parameters: {sum(p.numel() for p in state_decoder.parameters())}")
        
        # Mask decoder for selection mask prediction
        mask_decoder = MaskDecoder(
            image_size=image_size,
            latent_dim=latent_mask_dim,
            decoder_params=config.get('mask_decoder_params', {})
        ).to(device)
        print(f"[MaskDecoder] Number of parameters: {sum(p.numel() for p in mask_decoder.parameters())}")

    # Loss functions
    color_criterion = nn.CrossEntropyLoss()
    vicreg_loss_fn_selection = VICRegLoss(sim_coeff=vicreg_sim_coeff_selection, std_coeff=vicreg_std_coeff_selection, cov_coeff=vicreg_cov_coeff_selection)
    vicreg_loss_fn_next_state = VICRegLoss(sim_coeff=vicreg_sim_coeff_next_state, std_coeff=vicreg_std_coeff_next_state, cov_coeff=vicreg_cov_coeff_next_state)
    selection_criterion = nn.MSELoss()
    next_state_criterion = nn.MSELoss()
    reward_criterion = nn.L1Loss()
    
    # Optimize all modules together - now including both embedders
    if use_pretrained_encoder and freeze_pretrained_encoder:
        optimizer_params = (
            list(color_predictor.parameters()) + 
            list(mask_encoder.parameters()) + 
            list(selection_mask_predictor.parameters()) + 
            list(next_state_predictor.parameters()) +
            list(reward_predictor.parameters()) +
            list(colour_selection_embedder.parameters()) +
            list(selection_embedder.parameters())
        )
        if use_decoder_loss and state_decoder is not None and mask_decoder is not None:
            optimizer_params += list(state_decoder.parameters()) + list(mask_decoder.parameters())
        optimizer = optim.AdamW(optimizer_params, lr=learning_rate)
    else:
        optimizer_params = (
            list(state_encoder.parameters()) + 
            list(color_predictor.parameters()) + 
            list(mask_encoder.parameters()) + 
            list(selection_mask_predictor.parameters()) + 
            list(next_state_predictor.parameters()) +
            list(reward_predictor.parameters()) +
            list(colour_selection_embedder.parameters()) +
            list(selection_embedder.parameters())
        )
        if use_decoder_loss and state_decoder is not None and mask_decoder is not None:
            optimizer_params += list(state_decoder.parameters()) + list(mask_decoder.parameters())
        optimizer = optim.AdamW(optimizer_params, lr=learning_rate)

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
        reward_predictor.train()
        if use_decoder_loss:
            if state_decoder is not None:
                state_decoder.train()
            if mask_decoder is not None:
                mask_decoder.train()
        total_color_loss = 0
        total_selection_loss = 0
        total_next_state_loss = 0
        total_reward_loss = 0
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)):
            
            # State
            state = batch['state'].to(device)
            shape_h = batch.get('shape_h', None).to(device)
            shape_w = batch.get('shape_w', None).to(device)
            num_colors_grid = batch.get('num_colors_grid', None).to(device)
            most_present_color = batch.get('most_present_color', None).to(device)
            least_present_color = batch.get('least_present_color', None).to(device)

            # Next state
            next_state = batch['next_state'].to(device)
            shape_h_next = batch.get('shape_h_next', None).to(device)
            shape_w_next = batch.get('shape_w_next', None).to(device)
            num_colors_grid_next = batch.get('num_colors_grid_next', None).to(device)
            most_present_color_next = batch.get('most_present_color_next', None).to(device)
            least_present_color_next = batch.get('least_present_color_next', None).to(device)

            # Target state
            target_state = batch['target_state'].to(device)
            shape_h_target = batch.get('shape_h_target').to(device)
            shape_w_target = batch.get('shape_w_target').to(device)
            num_colors_grid_target = batch.get('num_colors_grid_target', None).to(device)
            most_present_color_target = batch.get('most_present_color_target', None).to(device)
            least_present_color_target = batch.get('least_present_color_target', None).to(device)

            # Actions
            action_colour = batch['action_colour'].to(device)
            action_selection = batch['action_selection'].to(device)
            action_transform = batch['action_transform'].to(device)
            
            # Outputs of actions
            selected_color = batch['colour'].to(device)
            selection_mask = batch['selection_mask'].to(device)
            reward = batch['reward'].to(device).to(device)

            if state.dim() == 3:
                state = state.unsqueeze(1)
                next_state = next_state.unsqueeze(1)
                selection_mask = selection_mask.unsqueeze(1)

            # Encode state and next_state
            if shape_h is not None:
                latent = state_encoder(state.to(torch.long), shape_h=shape_h.to(device), shape_w=shape_w.to(device), num_unique_colors=num_colors_grid.to(device), most_common_color=most_present_color.to(device), least_common_color=least_present_color.to(device))
                latent_next = target_encoder(next_state.to(torch.long), shape_h=shape_h_next, shape_w=shape_w_next, num_unique_colors=num_colors_grid_next, most_common_color=most_present_color_next, least_common_color=least_present_color_next)
            else:
                latent = state_encoder(state.to(torch.long))
                latent_next = target_encoder(next_state.to(torch.long))

            action_colour_onehot = one_hot(action_colour, num_color_selection_fns)
            action_selection_onehot = one_hot(action_selection, num_selection_fns)
            action_transform_onehot = one_hot(action_transform, num_transform_actions)

            # Color prediction - using embedded actions
            action_color_embedding = colour_selection_embedder(action_colour_onehot)
            color_logits = color_predictor(latent, action_color_embedding)
            color_loss = color_criterion(color_logits, selected_color)

            # Selection mask prediction - handle ground truth vs predicted inputs
            action_selection_embedding = selection_embedder(action_selection_onehot)
            
            # Ground truth switch: use one-hot encoded color vs predicted color distribution
            if use_ground_truth:
                # Use ground truth one-hot encoded color (always use num_arc_colors)
                color_input = one_hot(selected_color, num_arc_colors)
            else:
                # Use predicted color distribution
                color_input = color_logits.softmax(dim=1)
            
            pred_latent_mask = selection_mask_predictor(latent, action_selection_embedding, color_input)
            
            # Decoder switch: use decoder loss vs latent space loss
            if use_decoder_loss and mask_decoder is not None:
                # Decode predicted mask and compute loss against ground truth mask
                pred_mask_logits = mask_decoder(pred_latent_mask)
                # Compute cross-entropy loss on the decoded mask
                B, H, W, C = pred_mask_logits.shape
                mask_loss = F.cross_entropy(
                    pred_mask_logits.view(B*H*W, C),
                    selection_mask.view(B*H*W).long(),
                    reduction='mean'
                )
                selection_loss = mask_loss
            else:
                # Use latent space loss (original behavior)
                target_latent_mask = mask_encoder(selection_mask.to(torch.long))
                if use_vicreg_selection:
                    selection_loss, _, _, _ = vicreg_loss_fn_selection(pred_latent_mask, target_latent_mask)
                else:
                    selection_loss = selection_criterion(pred_latent_mask, target_latent_mask)

            # Next state prediction - handle ground truth vs predicted inputs
            # Ground truth switch: use encoded ground truth mask vs predicted mask
            if use_ground_truth:
                # Use ground truth encoded mask
                mask_input = mask_encoder(selection_mask.to(torch.long))
            else:
                # Use predicted mask
                mask_input = pred_latent_mask
            
            pred_next_latent = next_state_predictor(latent, action_transform_onehot, mask_input)
            
            # Decoder switch: use decoder loss vs latent space loss
            if use_decoder_loss and state_decoder is not None:
                # Decode predicted next state and compute loss against ground truth next state
                pred_next_state_logits = state_decoder(pred_next_latent)
                # Compute autoencoder-style loss
                next_state_loss, _ = autoencoder_loss(
                    pred_next_state_logits, 
                    next_state, 
                    shape_h_next, 
                    shape_w_next, 
                    most_present_color_next, 
                    least_present_color_next, 
                    num_colors_grid_next
                )
            else:
                # Use latent space loss (original behavior)
                if use_vicreg_next_state:
                    next_state_loss, _, _, _ = vicreg_loss_fn_next_state(pred_next_latent, latent_next)
                else:
                    next_state_loss = next_state_criterion(pred_next_latent, latent_next)

            # Reward prediction - handle ground truth vs predicted inputs
            # Ground truth switch: use ground truth next state vs predicted next state
            if use_ground_truth:
                # Use ground truth next state
                next_state_input = latent_next
            else:
                # Use predicted next state
                next_state_input = pred_next_latent
            
            # Encode target state for reward prediction
            latent_target = target_encoder(target_state, shape_h=shape_h_target, shape_w=shape_w_target, num_unique_colors=num_colors_grid_target, most_common_color=most_present_color_target, least_common_color=least_present_color_target)
            
            pred_reward = reward_predictor(latent, next_state_input, latent_target)
            reward_loss = reward_criterion(pred_reward.squeeze(-1), reward.float())

            # Combined loss
            total_loss = color_loss + selection_loss + next_state_loss + reward_loss

            optimizer.zero_grad()
            total_loss.backward()
            grad_params = (
                list(state_encoder.parameters()) + 
                list(color_predictor.parameters()) + 
                list(mask_encoder.parameters()) + 
                list(selection_mask_predictor.parameters()) +
                list(colour_selection_embedder.parameters()) +
                list(selection_embedder.parameters()) +
                list(reward_predictor.parameters())
            )
            if use_decoder_loss and state_decoder is not None and mask_decoder is not None:
                grad_params += list(state_decoder.parameters()) + list(mask_decoder.parameters())
            torch.nn.utils.clip_grad_norm_(grad_params, max_norm=1.0)
            optimizer.step()

            # EMA update for target encoder
            update_ema(target_encoder, state_encoder, decay=0.995)

            total_color_loss += color_loss.item() * state.size(0)
            total_selection_loss += selection_loss.item() * state.size(0)
            total_next_state_loss += next_state_loss.item() * state.size(0)
            total_reward_loss += reward_loss.item() * state.size(0)

            if (i + 1) % log_interval == 0:
                wandb.log({  # type: ignore
                    "batch_color_loss": color_loss.item(),
                    "batch_selection_loss": selection_loss.item(),
                    "batch_next_state_loss": next_state_loss.item(),
                    "batch_reward_loss": reward_loss.item(),
                    "batch_total_loss": total_loss.item(),
                    "epoch": epoch + 1,
                    "batch": i + 1
                })

        num_train_samples = sum(batch['state'].size(0) for batch in train_loader)
        avg_color_loss = total_color_loss / num_train_samples
        avg_selection_loss = total_selection_loss / num_train_samples
        avg_next_state_loss = total_next_state_loss / num_train_samples
        avg_reward_loss = total_reward_loss / num_train_samples
        avg_total_loss = avg_color_loss + avg_selection_loss + avg_next_state_loss + avg_reward_loss

        val_color_loss, val_selection_loss, val_next_state_loss, val_color_acc, val_color_class_acc, val_next_state_cosine = evaluate_all_modules(
            color_predictor, selection_mask_predictor, next_state_predictor, state_encoder, target_encoder, mask_encoder,
            colour_selection_embedder, selection_embedder, val_loader, device, color_criterion, num_color_selection_fns, num_selection_fns, num_transform_actions,
            use_vicreg_selection, vicreg_loss_fn_selection, selection_criterion, use_vicreg_next_state, vicreg_loss_fn_next_state, next_state_criterion,
            state_decoder, mask_decoder, use_ground_truth, use_decoder_loss, num_arc_colors
        )

        # --- Compute validation reward loss --- # TO REVIEW IS THIS CORRECT
        reward_predictor.eval()
        total_val_reward_loss = 0
        total_val_samples = 0
        with torch.no_grad():
            for batch in val_loader:
                state = batch['state'].to(device)
                next_state = batch['next_state'].to(device)
                target_state = batch['target_state'].to(device)
                reward = batch['reward'].to(device)
                shape_h = batch.get('shape_h', None)
                shape_w = batch.get('shape_w', None)
                num_colors_grid = batch.get('num_colors_grid', None)
                most_present_color = batch.get('most_present_color', None)
                least_present_color = batch.get('least_present_color', None)

                if state.dim() == 3:
                    state = state.unsqueeze(1)
                    next_state = next_state.unsqueeze(1)

                # Encode state and next_state
                if shape_h is not None:
                    latent = state_encoder(state.to(torch.long), shape_h=shape_h.to(device), shape_w=shape_w.to(device), num_unique_colors=num_colors_grid.to(device), most_common_color=most_present_color.to(device), least_common_color=least_present_color.to(device))
                    latent_next = target_encoder(next_state.to(torch.long), shape_h=shape_h.to(device), shape_w=shape_w.to(device), num_unique_colors=num_colors_grid.to(device), most_common_color=most_present_color.to(device), least_common_color=least_present_color.to(device))
                    latent_target = target_encoder(target_state.to(torch.long), shape_h=shape_h.to(device), shape_w=shape_w.to(device), num_unique_colors=num_colors_grid.to(device), most_common_color=most_present_color.to(device), least_common_color=least_present_color.to(device))
                else:
                    latent = state_encoder(state.to(torch.long))
                    latent_next = target_encoder(next_state.to(torch.long))
                    latent_target = target_encoder(target_state.to(torch.long))

                pred_reward = reward_predictor(latent, latent_next, latent_target)
                reward_loss = reward_criterion(pred_reward.squeeze(-1), reward.float())
                total_val_reward_loss += reward_loss.item() * state.size(0)
                total_val_samples += state.size(0)
        val_reward_loss = total_val_reward_loss / total_val_samples if total_val_samples > 0 else 0.0

        val_total_loss = val_color_loss + val_selection_loss + val_next_state_loss + val_reward_loss
        print(f"Epoch {epoch+1}/{num_epochs} - Train Color Loss: {avg_color_loss:.4f} | Train Selection Loss: {avg_selection_loss:.4f} | Train Next State Loss: {avg_next_state_loss:.4f} | Train Reward Loss: {avg_reward_loss:.4f} | Train Total Loss: {avg_total_loss:.4f} | Val Color Loss: {val_color_loss:.4f} | Val Selection Loss: {val_selection_loss:.4f} | Val Next State Loss: {val_next_state_loss:.4f} | Val Reward Loss: {val_reward_loss:.4f} | Val Color Acc: {val_color_acc:.4f} | Val Total Loss: {val_total_loss:.4f}")
        # --- WANDB LOGGING FOR EPOCH ---
        wandb.log({  # type: ignore
            "epoch": epoch + 1,
            "train_color_loss": avg_color_loss,
            "train_selection_loss": avg_selection_loss,
            "train_next_state_loss": avg_next_state_loss,
            "train_reward_loss": avg_reward_loss,
            "train_total_loss": avg_total_loss,
            "val_color_loss": val_color_loss,
            "val_selection_loss": val_selection_loss,
            "val_next_state_loss": val_next_state_loss,
            "val_reward_loss": val_reward_loss,
            "val_color_acc": val_color_acc,
            "val_next_state_cosine": val_next_state_cosine,
            "val_total_loss": val_total_loss
        })

        # Log per-class accuracies separately
        if val_color_class_acc is not None:
            for i, acc in enumerate(val_color_class_acc):
                wandb.log({f"val_color_class_{i}_acc": acc})  # type: ignore

        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            epochs_no_improve = 0
            save_dict = {
                'state_encoder': state_encoder.state_dict(),
                'color_predictor': color_predictor.state_dict(),
                'mask_encoder': mask_encoder.state_dict(),
                'selection_mask_predictor': selection_mask_predictor.state_dict(),
                'next_state_predictor': next_state_predictor.state_dict(),
                'reward_predictor': reward_predictor.state_dict(),
                'colour_selection_embedder': colour_selection_embedder.state_dict(),
                'selection_embedder': selection_embedder.state_dict(),
                'target_encoder': target_encoder.state_dict()
            }
            if use_decoder_loss and state_decoder is not None and mask_decoder is not None:
                save_dict.update({
                    'state_decoder': state_decoder.state_dict(),
                    'mask_decoder': mask_decoder.state_dict()
                })
            torch.save(save_dict, save_path)
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