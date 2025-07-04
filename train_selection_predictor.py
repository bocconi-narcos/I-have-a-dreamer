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
from src.models.base.transformer_blocks import Transformer

# --- Config Loader ---
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# --- One-hot encoding utility ---
def one_hot(indices, num_classes):
    return torch.nn.functional.one_hot(indices, num_classes=num_classes).float()

# --- Validation Metrics ---
def evaluate_selection_and_color(selection_predictor, color_predictor, state_encoder, mask_encoder, 
                                dataloader, device, criterion, num_color_selection_fns, num_selection_fns, use_vicreg, vicreg_loss_fn=None):
    selection_predictor.eval()
    color_predictor.eval()
    state_encoder.eval()
    mask_encoder.eval()
    total_selection_loss = 0
    total_color_loss = 0
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
            
            print("State min:", state.min().item(), "max:", state.max().item())
            
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
            
            # Color prediction
            color_input = torch.cat([latent, action_colour_onehot], dim=1)
            color_logits = color_predictor(color_input)
            color_loss = criterion(color_logits, target_colour)
            
            # Selection mask prediction - updated to use new signature
            pred_latent_mask = selection_predictor(latent, action_selection_onehot, color_logits.softmax(dim=1))
            target_latent_mask = mask_encoder(selection_mask.float())
            
            if use_vicreg and vicreg_loss_fn is not None:
                selection_loss, _, _, _ = vicreg_loss_fn(pred_latent_mask, target_latent_mask)
            else:
                selection_loss = nn.MSELoss()(pred_latent_mask, target_latent_mask)
            
            total_selection_loss += selection_loss.item() * state.size(0)
            total_color_loss += color_loss.item() * state.size(0)
            color_preds = torch.argmax(color_logits, dim=1)
            color_correct += (color_preds == target_colour).sum().item()
            total += state.size(0)
    
    avg_selection_loss = total_selection_loss / total
    avg_color_loss = total_color_loss / total
    color_accuracy = color_correct / total
    return avg_selection_loss, avg_color_loss, color_accuracy

def action_selection_embedder():
    
    
def action_color_embedder():
    
    

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
    encoder_type = config['encoder_type']
    latent_dim = config['latent_dim']
    encoder_params = config['encoder_params']
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
    transformer_depth = selection_cfg['transformer_depth']
    transformer_heads = selection_cfg['transformer_heads']
    transformer_dim_head = selection_cfg['transformer_dim_head']
    transformer_mlp_dim = selection_cfg['transformer_mlp_dim']
    transformer_dropout = selection_cfg['transformer_dropout']
    use_vicreg = selection_cfg['use_vicreg']
    vicreg_sim_coeff = selection_cfg['vicreg_sim_coeff']
    vicreg_std_coeff = selection_cfg['vicreg_std_coeff']
    vicreg_cov_coeff = selection_cfg['vicreg_cov_coeff']
    
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_encoder = StateEncoder(
        image_size=image_size,
        input_channels=input_channels,
        latent_dim=latent_dim,
        encoder_params=encoder_params
    ).to(device)
    color_predictor = ColorPredictor(latent_dim + num_color_selection_fns, num_arc_colors, color_predictor_hidden_dim).to(device)
    # mask_encoder = MaskEncoder(mask_encoder_type, **mask_encoder_params).to(device)
    mask_encoder = MaskEncoder(
    image_size=mask_encoder_params['image_size'],
    vocab_size=mask_encoder_params['vocab_size'],
    emb_dim=mask_encoder_params.get('emb_dim', 64),
    depth=mask_encoder_params.get('depth', 4),
    heads=mask_encoder_params.get('heads', 8),
    mlp_dim=mask_encoder_params.get('mlp_dim', 512),
    dropout=mask_encoder_params.get('dropout', 0.2),
    emb_dropout=mask_encoder_params.get('emb_dropout', 0.2),
    padding_value=mask_encoder_params.get('padding_value', -1)
    ).to(device)
    
    # Updated SelectionMaskPredictor initialization
    selection_mask_predictor = SelectionMaskPredictor(
        state_dim=latent_dim,
        selection_action_dim=num_selection_fns,
        color_pred_dim=num_arc_colors,  # Assuming color prediction dimension
        latent_mask_dim=latent_mask_dim,
        transformer_depth=transformer_depth,
        transformer_heads=transformer_heads,
        transformer_dim_head=transformer_dim_head,
        transformer_mlp_dim=transformer_mlp_dim,
        dropout=transformer_dropout
    ).to(device)
    
    # Create a color selection embedder and action selection embedder, and add in optimizer these parameters
    # Color predictor takes the embedded action, learnable embedding, so actions and subactions are not one hot encoded but latent vectors. 
    # Take state, action_colour etc, embed state, embed actions, pass color predictor, finally get logits.
    
    color_criterion = nn.CrossEntropyLoss()
    selection_criterion = nn.MSELoss()
    # vicreg_loss_fn = VICRegLoss(sim_coeff=vicreg_sim_coeff, std_coeff=vicreg_std_coeff, cov_coeff=vicreg_cov_coeff)
    
    # Optimize all modules together
    optimizer = optim.AdamW(
        list(state_encoder.parameters()) + 
        list(color_predictor.parameters()) + 
        list(mask_encoder.parameters()) + 
        list(selection_mask_predictor.parameters()), 
        lr=learning_rate
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 10
    save_path = 'best_model_selection_predictor.pth'
    for epoch in range(num_epochs):
        state_encoder.train()
        color_predictor.train()
        mask_encoder.train()
        selection_mask_predictor.train()
        total_selection_loss = 0
        total_color_loss = 0
        
        for i, batch in enumerate(train_loader):
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
            
            print("State min:", state.min().item(), "max:", state.max().item())
            
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
            
            # Color prediction
            action_color_embedding = 
            color_logits = color_predictor(color_input)
            color_loss = color_criterion(color_logits, target_colour)
            
            # Selection mask prediction - updated to use new signature
            pred_latent_mask = selection_mask_predictor(latent, action_selection_onehot, color_logits.softmax(dim=1))
            target_latent_mask = mask_encoder(selection_mask.float())
            
            if use_vicreg:
                selection_loss, _, _, _ = vicreg_loss_fn(pred_latent_mask, target_latent_mask)
            else:
                selection_loss = selection_criterion(pred_latent_mask, target_latent_mask)
            
            # Combined loss
            total_loss = color_loss + selection_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            total_selection_loss += selection_loss.item() * state.size(0)
            total_color_loss += color_loss.item() * state.size(0)
            
            if (i + 1) % log_interval == 0:
                print(f"Epoch {epoch+1} Batch {i+1}/{len(train_loader)} - Color Loss: {color_loss.item():.4f} | Selection Loss: {selection_loss.item():.4f}")

        avg_selection_loss, avg_color_loss, color_accuracy = evaluate_selection_and_color(
            selection_mask_predictor, color_predictor, state_encoder, mask_encoder, val_loader, device, color_criterion, num_color_selection_fns, num_selection_fns, use_vicreg, vicreg_loss_fn)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Selection Loss: {total_selection_loss/len(train_loader.dataset):.4f} | Train Color Loss: {total_color_loss/len(train_loader.dataset):.4f} | Val Selection Loss: {avg_selection_loss:.4f} | Val Color Loss: {avg_color_loss:.4f} | Val Color Acc: {color_accuracy:.4f}")
        if avg_selection_loss < best_val_loss:
            best_val_loss = avg_selection_loss
            epochs_no_improve = 0
            torch.save({
                'state_encoder': state_encoder.state_dict(),
                'color_predictor': color_predictor.state_dict(),
                'mask_encoder': mask_encoder.state_dict(),
                'selection_mask_predictor': selection_mask_predictor.state_dict()
            }, save_path)
            print(f"New best model saved to {save_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s)")
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss for {patience} epochs.")
            break

if __name__ == "__main__":
    train_selection_predictor() 