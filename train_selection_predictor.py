import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
from src.models.state_encoder import StateEncoder
from src.models.predictors.color_predictor import ColorPredictor
from src.models.mask_encoder_new import MaskEncoder
from src.models.predictors.selection_mask_predictor import SelectionMaskPredictor
from src.losses.vicreg import VICRegLoss
from src.data import ReplayBufferDataset
from src.models.action_embed import ActionEmbedder
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Warning: wandb not available. Training will continue without logging.")
    WANDB_AVAILABLE = False
from tqdm import tqdm

# --- Config Loader ---
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# --- One-hot encoding utility ---
def one_hot(indices, num_classes):
    return torch.nn.functional.one_hot(indices, num_classes=num_classes).float()

# --- Validation Metrics ---
def evaluate_selection_and_color(selection_predictor, color_predictor, state_encoder, mask_encoder, 
                                colour_selection_embedder, selection_embedder,
                                dataloader, device, criterion, num_color_selection_fns, num_selection_fns, vicreg_loss_fn, mse_loss_fn, use_vicreg):
    selection_predictor.eval()
    color_predictor.eval()
    state_encoder.eval()
    mask_encoder.eval()
    total_selection_loss = 0
    total_color_loss = 0
    total_sim_loss = 0
    total_std_loss = 0
    total_cov_loss = 0
    color_correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            state = batch['state'].to(device)
            if state.dim() == 4 and state.shape[1] == 1:
                state = state.squeeze(1)
            state = state.long()
            action_colour = batch['action_colour'].to(device)
            action_selection = batch['action_selection'].to(device)
            target_colour = batch['colour'].to(device)
            selection_mask = batch['selection_mask'].to(device)
            shape_w = batch['shape_w'].to(device)
            shape_h = batch['shape_h'].to(device)
            num_colors_grid = batch['num_colors_grid'].to(device)
            most_present_color = batch['most_present_color'].to(device)
            least_present_color = batch['least_present_color'].to(device)
            
            if state.dim() == 3:
                state = state.unsqueeze(1)
                selection_mask = selection_mask.unsqueeze(1)
            
            # print("State min:", state.min().item(), "max:", state.max().item())
            
            latent = state_encoder(
                state,
                shape_w=shape_w,
                shape_h=shape_h,
                num_unique_colors=num_colors_grid,
                most_common_color=most_present_color,
                least_common_color=least_present_color
            )
            action_colour_onehot = one_hot(action_colour, num_color_selection_fns)
            action_selection_onehot = one_hot(action_selection, num_selection_fns)
            
            # Color prediction - using embedded actions
            action_color_embedding = colour_selection_embedder(action_colour_onehot)
            color_logits = color_predictor(latent, action_color_embedding)
            color_loss = criterion(color_logits, target_colour)
            
            # Selection mask prediction - now using embedded selection actions
            action_selection_embedding = selection_embedder(action_selection_onehot)
            pred_latent_mask = selection_predictor(latent, action_selection_embedding, color_logits.softmax(dim=1))
            target_latent_mask = mask_encoder(selection_mask.long())
            
            # Use VICReg or MSE loss for selection mask prediction
            if use_vicreg:
                selection_loss, sim_loss, std_loss, cov_loss = vicreg_loss_fn(pred_latent_mask, target_latent_mask)
            else:
                selection_loss = mse_loss_fn(pred_latent_mask, target_latent_mask)
                sim_loss = torch.tensor(0.0)
                std_loss = torch.tensor(0.0)
                cov_loss = torch.tensor(0.0)
            
            total_selection_loss += selection_loss.item() * state.size(0)
            total_color_loss += color_loss.item() * state.size(0)
            total_sim_loss += sim_loss.item() * state.size(0)
            total_std_loss += std_loss.item() * state.size(0)
            total_cov_loss += cov_loss.item() * state.size(0)
            color_preds = torch.argmax(color_logits, dim=1)
            color_correct += (color_preds == target_colour).sum().item()
            total += state.size(0)
    
    avg_selection_loss = total_selection_loss / total
    avg_color_loss = total_color_loss / total
    avg_sim_loss = total_sim_loss / total
    avg_std_loss = total_std_loss / total
    avg_cov_loss = total_cov_loss / total
    color_accuracy = color_correct / total
    return avg_selection_loss, avg_color_loss, color_accuracy, avg_sim_loss, avg_std_loss, avg_cov_loss    
    

# --- Main Training Loop ---
def train_selection_predictor():
    """
    Main training loop for both selection mask predictor and color predictor. Loads config, prepares dataset, builds models, and trains.
    The buffer is expected to be a list of dicts with the required keys. The training loop:
        1. Extracts action['selection'] and action['colour'], one-hot encodes them.
        2. Passes state through the configurable state encoder.
        3. For selection: concatenates state embedding and selection action encoding, passes through selection mask predictor.
        4. For selection: passes ground truth selection_mask through mask encoder to get target latent mask.
        5. For color: concatenates state embedding and color action encoding, passes through color predictor MLP.
        6. Computes MSE/VICReg loss for selection and cross-entropy loss for color.
    All model choices and hyperparameters are loaded from config.yaml.
    """
    config = load_config()
    buffer_path = config['buffer_path']
    latent_dim = config['latent_dim']

    # Encoder parameters
    encoder_params = config['encoder_params']
    mask_encoder_params = config['selection_mask']['mask_encoder_params']

    # Mask Encoder parameters
    latent_mask_dim = config['selection_mask']['latent_mask_dim']
    mask_encoder_depth = mask_encoder_params['depth']
    mask_encoder_heads = mask_encoder_params['heads']
    mask_encoder_mlp_dim = mask_encoder_params['mlp_dim']
    mask_encoder_dropout = mask_encoder_params['dropout']
    mask_encoder_vocab_size = mask_encoder_params['vocab_size']

    # Color predictor parameters
    mask_predictor_params = config['selection_mask']['mask_predictor_params']
    transformer_depth = mask_predictor_params['transformer_depth']
    transformer_heads = mask_predictor_params['transformer_heads']
    transformer_dim_head = mask_predictor_params['transformer_dim_head']
    transformer_mlp_dim = mask_predictor_params['transformer_mlp_dim']
    transformer_dropout = mask_predictor_params['transformer_dropout']

    # Action embedder parameters
    num_color_selection_fns = config['action_embedders']['action_color_embedder']['num_actions']
    num_selection_fns = config['action_embedders']['action_selection_embedder']['num_actions']
    num_transform_actions = config['action_embedders']['action_transform_embedder']['num_actions']
    color_selection_dim = config['action_embedders']['action_color_embedder']['embed_dim']
    selection_dim = config['action_embedders']['action_selection_embedder']['embed_dim']
    
    # Color predictor parameters
    num_arc_colors = config['num_arc_colors']
    color_predictor_hidden_dim = config['color_predictor']['hidden_dim']

    # VICReg parameters
    selection_cfg = config['selection_mask']
    use_vicreg = selection_cfg.get('use_vicreg', True)
    vicreg_sim_coeff = selection_cfg['vicreg_sim_coeff']
    vicreg_std_coeff = selection_cfg['vicreg_std_coeff']
    vicreg_cov_coeff = selection_cfg['vicreg_cov_coeff']
    
    # Training parameters
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
        mode='selection_color'
    )
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Initialize models
    state_encoder = StateEncoder(
        image_size=image_size,
        input_channels=input_channels,
        latent_dim=latent_dim,
        encoder_params=encoder_params
    ).to(device)
    color_predictor = ColorPredictor(latent_dim=latent_dim, 
                                     num_colors=num_arc_colors - 1, 
                                     hidden_dim=color_predictor_hidden_dim,
                                     action_embedding_dim=color_selection_dim).to(device)

    # mask_encoder = MaskEncoder(mask_encoder_type, **mask_encoder_params).to(device)
    mask_encoder = MaskEncoder(
    image_size=image_size,
    vocab_size=mask_encoder_vocab_size,
    emb_dim=latent_mask_dim,
    depth=mask_encoder_depth,
    heads=mask_encoder_heads,
    mlp_dim=mask_encoder_mlp_dim,
    dropout=mask_encoder_dropout,
    ).to(device)
    
    # Updated SelectionMaskPredictor initialization - now takes embedded selection actions
    selection_mask_predictor = SelectionMaskPredictor(
        state_dim=latent_dim,
        selection_action_embed_dim=selection_dim,  # Use embedded dimension instead of one-hot
        color_pred_dim=num_arc_colors-1,  # Assuming color prediction dimension
        latent_mask_dim=latent_mask_dim,
        transformer_depth=transformer_depth,
        transformer_heads=transformer_heads,
        transformer_dim_head=transformer_dim_head,
        transformer_mlp_dim=transformer_mlp_dim,
        dropout=transformer_dropout
    ).to(device)
    
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

    
    # Create a color selection embedder and action selection embedder, and add in optimizer these parameters
    # Color predictor takes the embedded action, learnable embedding, so actions and subactions are not one hot encoded but latent vectors. 
    # Take state, action_colour etc, embed state, embed actions, pass color predictor, finally get logits.
    
    color_criterion = nn.CrossEntropyLoss()
    vicreg_loss_fn = VICRegLoss(sim_coeff=vicreg_sim_coeff, std_coeff=vicreg_std_coeff, cov_coeff=vicreg_cov_coeff)
    mse_loss_fn = nn.MSELoss()
    
    # Optimize all modules together - now including both embedders
    optimizer = optim.AdamW(
        list(state_encoder.parameters()) + 
        list(color_predictor.parameters()) + 
        list(mask_encoder.parameters()) + 
        list(selection_mask_predictor.parameters()) +
        list(colour_selection_embedder.parameters()) +
        list(selection_embedder.parameters()), 
        lr=learning_rate
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10
    save_path = 'best_model_selection_predictor.pth'
    if WANDB_AVAILABLE:
        wandb.init(project="selection_predictor", config=config)
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print("-" * 80)
    
    # Main epoch progress bar
    epoch_pbar = tqdm(range(num_epochs), desc='Training Progress', ncols=120)
    for epoch in epoch_pbar:
        state_encoder.train()
        color_predictor.train()
        mask_encoder.train()
        selection_mask_predictor.train()
        total_selection_loss = 0
        total_color_loss = 0
        total_sim_loss = 0
        total_std_loss = 0
        total_cov_loss = 0
        
        # Create progress bar for training batches
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', 
                         leave=False, ncols=100)
        
        for i, batch in enumerate(train_pbar):
            state = batch['state'].to(device)
            if state.dim() == 4 and state.shape[1] == 1:
                state = state.squeeze(1)
            state = state.long()
            action_colour = batch['action_colour'].to(device)
            action_selection = batch['action_selection'].to(device)
            target_colour = batch['colour'].to(device)
            selection_mask = batch['selection_mask'].to(device)
            shape_w = batch['shape_w'].to(device)
            shape_h = batch['shape_h'].to(device)
            num_colors_grid = batch['num_colors_grid'].to(device)
            most_present_color = batch['most_present_color'].to(device)
            least_present_color = batch['least_present_color'].to(device)
            
            if state.dim() == 3:
                state = state.unsqueeze(1)
                selection_mask = selection_mask.unsqueeze(1)
                        
            latent = state_encoder(
                state,
                shape_w=shape_w,
                shape_h=shape_h,
                num_unique_colors=num_colors_grid,
                most_common_color=most_present_color,
                least_common_color=least_present_color
            )
            action_colour_onehot = one_hot(action_colour, num_color_selection_fns).to(device)
            action_selection_onehot = one_hot(action_selection, num_selection_fns).to(device)
            
            # Color prediction
            action_color_embedding = colour_selection_embedder(action_colour_onehot)
            #action_selection_embedding = selection_embedder(action_selection_onehot)

            # Calculate color logits and loss
            color_logits = color_predictor(latent, action_color_embedding)
            color_loss = color_criterion(color_logits, target_colour)

            # Selection mask prediction - now using embedded selection actions
            action_selection_embedding = selection_embedder(action_selection_onehot)

            pred_latent_mask = selection_mask_predictor(latent, action_selection_embedding, color_logits.softmax(dim=1))
            target_latent_mask = mask_encoder(selection_mask.long())
            
            # Use VICReg or MSE loss for selection mask prediction
            if use_vicreg:
                selection_loss, sim_loss, std_loss, cov_loss = vicreg_loss_fn(pred_latent_mask, target_latent_mask)
            else:
                selection_loss = mse_loss_fn(pred_latent_mask, target_latent_mask)
                sim_loss = torch.tensor(0.0)
                std_loss = torch.tensor(0.0)
                cov_loss = torch.tensor(0.0)
            
            # Combined loss
            total_loss = color_loss + selection_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(state_encoder.parameters()) + 
                list(color_predictor.parameters()) + 
                list(mask_encoder.parameters()) + 
                list(selection_mask_predictor.parameters()) +
                list(colour_selection_embedder.parameters()) +
                list(selection_embedder.parameters()), 
                max_norm=1.0
            )
            optimizer.step()

            # Log gradients to wandb
            if WANDB_AVAILABLE and (i % log_interval == 0):
                # Log training metrics
                step = epoch * len(train_loader) + i
                wandb.log({
                    'batch/color_loss': color_loss.item(),
                    'batch/selection_loss': selection_loss.item(),
                    'batch/vicreg_sim_loss': sim_loss.item(),
                    'batch/vicreg_std_loss': std_loss.item(),
                    'batch/vicreg_cov_loss': cov_loss.item(),
                    'batch/total_loss': total_loss.item(),
                }, step=step)
                
             
            total_selection_loss += selection_loss.item() * state.size(0)
            total_color_loss += color_loss.item() * state.size(0)
            total_sim_loss += sim_loss.item() * state.size(0)
            total_std_loss += std_loss.item() * state.size(0)
            total_cov_loss += cov_loss.item() * state.size(0)
            
            # Update progress bar with current loss
            train_pbar.set_postfix({'Color Loss': f'{color_loss.item():.4f}', 'Selection Loss': f'{selection_loss.item():.4f}'})

        train_pbar.close()
        
        avg_selection_loss, avg_color_loss, color_accuracy, avg_val_sim_loss, avg_val_std_loss, avg_val_cov_loss = evaluate_selection_and_color(
            selection_mask_predictor, color_predictor, state_encoder, mask_encoder, 
            colour_selection_embedder, selection_embedder,
            val_loader, device, color_criterion, num_color_selection_fns, num_selection_fns, vicreg_loss_fn, mse_loss_fn, use_vicreg)
        
        # Calculate average training losses
        num_train_samples = sum(batch['state'].size(0) for batch in train_loader)
        avg_train_selection_loss = total_selection_loss / num_train_samples
        avg_train_color_loss = total_color_loss / num_train_samples
        avg_train_sim_loss = total_sim_loss / num_train_samples
        avg_train_std_loss = total_std_loss / num_train_samples
        avg_train_cov_loss = total_cov_loss / num_train_samples
        
        # Log to wandb
        if WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch + 1,
                'train/selection_loss': avg_train_selection_loss,
                'train/color_loss': avg_train_color_loss,
                'train/vicreg_sim_loss': avg_train_sim_loss,
                'train/vicreg_std_loss': avg_train_std_loss,
                'train/vicreg_cov_loss': avg_train_cov_loss,
                'val/selection_loss': avg_selection_loss,
                'val/color_loss': avg_color_loss,
                'val/color_accuracy': color_accuracy,
                'val/vicreg_sim_loss': avg_val_sim_loss,
                'val/vicreg_std_loss': avg_val_std_loss,
                'val/vicreg_cov_loss': avg_val_cov_loss
            })
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'Train Sel': f'{avg_train_selection_loss:.4f}',
            'Train Col': f'{avg_train_color_loss:.4f}',
            'Val Sel': f'{avg_selection_loss:.4f}',
            'Val Col': f'{avg_color_loss:.4f}',
            'Val Acc': f'{color_accuracy:.4f}',
            'Best': f'{best_val_loss:.4f}'
        })
        # Check for improvement
        improvement_status = ""
        if avg_selection_loss < best_val_loss:
            best_val_loss = avg_selection_loss
            epochs_no_improve = 0
            torch.save({
                'state_encoder': state_encoder.state_dict(),
                'color_predictor': color_predictor.state_dict(),
                'mask_encoder': mask_encoder.state_dict(),
                'selection_mask_predictor': selection_mask_predictor.state_dict(),
                'colour_selection_embedder': colour_selection_embedder.state_dict(),
                'selection_embedder': selection_embedder.state_dict()
            }, save_path)
            torch.save(state_encoder.state_dict(), 'best_model_state_encoder.pth')
            improvement_status = f" ✓ New best model saved!"
        else:
            epochs_no_improve += 1
            improvement_status = f" ({epochs_no_improve} epochs without improvement)"
        
        # Print detailed epoch summary
        tqdm.write(f"Epoch {epoch+1:3d}/{num_epochs} | Train Sel: {avg_train_selection_loss:.4f} | Train Col: {avg_train_color_loss:.4f} | Val Sel: {avg_selection_loss:.4f} | Val Col: {avg_color_loss:.4f} | Acc: {color_accuracy:.4f}{improvement_status}")
        
        # Early stopping check
        if epochs_no_improve >= patience:
            tqdm.write(f"\nEarly stopping at epoch {epoch+1} due to no improvement for {patience} epochs.")
            break

    epoch_pbar.close()
    
    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best validation selection loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {save_path}")
    print("="*80)

    if WANDB_AVAILABLE:
        wandb.finish()

if __name__ == "__main__":
    train_selection_predictor() 