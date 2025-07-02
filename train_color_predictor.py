import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
import pickle
from src.models.state_encoder import StateEncoder
from src.models.predictors.color_predictor import ColorPredictor, TransformerColorPredictor
from src.data import ReplayBufferDataset
from torch.utils.data import Dataset
from src.models.action_embed import ActionEmbedder
import wandb
from tqdm import tqdm

# --- Config Loader ---
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# --- One-hot encoding utility ---
def one_hot(indices, num_classes):
    return torch.nn.functional.one_hot(indices, num_classes=num_classes).float()

# --- Validation Metrics ---
def evaluate(model, encoder, action_embedder, dataloader, device, criterion, num_color_selection_fns):
    model.eval()
    encoder.eval()
    action_embedder.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        # Create progress bar for validation
        val_pbar = tqdm(dataloader, desc='Validating', leave=False, ncols=80)
        
        for batch in val_pbar:
            state                           = batch['state'].to(device)
            action_colour                   = batch['action_colour'].to(device)
            target_colour                   = batch['colour'].to(device)
            shape_w                         = batch['shape_w'].to(device)
            shape_h                         = batch['shape_h'].to(device)
            num_colors_grid                 = batch['num_colors_grid'].to(device)
            most_present_color              = batch['most_present_color'].to(device)
            least_present_color             = batch['least_present_color'].to(device)
            
            latent                          = encoder(
                state,
                shape_w=shape_w,
                shape_h=shape_h,
                num_unique_colors=num_colors_grid,
                most_common_color=most_present_color,
                least_common_color=least_present_color
            )
            action_colour_onehot            = one_hot(action_colour, num_color_selection_fns)
            action_embedding                = action_embedder(action_colour_onehot)
            logits                          = model(latent, action_embedding)
            loss                            = criterion(logits, target_colour)
            total_loss                     += loss.item() * state.size(0)
            preds                           = torch.argmax(logits, dim=1)
            correct                        += (preds == target_colour).sum().item()
            total                          += state.size(0)
            
            # Update progress bar with current metrics
            current_acc = correct / total if total > 0 else 0
            val_pbar.set_postfix({'Acc': f'{current_acc:.3f}'})
        
        val_pbar.close()
        
    avg_loss                                = total_loss / total
    accuracy                                = correct / total
    return avg_loss, accuracy




# --- Main Training Loop ---
def train_color_predictor():
    """
    Main training loop for the color predictor. Loads config, prepares dataset, builds models, and trains.
    The buffer is expected to be a list of dicts with the required keys. The training loop:
        1. Extracts action['colour'] and one-hot encodes it.
        2. Passes state through the configurable state encoder.
        3. Concatenates state embedding and color action encoding, passes through MLP.
        4. Computes cross-entropy loss with the true colour.
    All model choices and hyperparameters are loaded from unified_config.yaml.
    """
    config                                  = load_config()
    buffer_path                             = config['buffer_path']
    #encoder_type = config['encoder_type']
    latent_dim                              = config['latent_dim']
    
    encoder_params                          = config['encoder_params']
    use_pretrained_encoder                  = config.get('use_pretrained_encoder', False)
    pretrained_encoder_path                 = config.get('pretrained_encoder_path', 'best_model_autoencoder.pth')
    freeze_pretrained_encoder               = config.get('freeze_pretrained_encoder', False)
    num_color_selection_fns                 = config['num_color_selection_fns']
    num_selection_fns                       = config['num_selection_fns']
    num_transform_actions                   = config['num_transform_actions']
    num_arc_colors                          = config['num_arc_colors']
    color_predictor_hidden_dim              = config['color_predictor']['hidden_dim']
    batch_size                              = config['batch_size']
    num_epochs                              = config['num_epochs']
    learning_rate                           = config['learning_rate']
    num_workers                             = config['num_workers']
    log_interval                            = config['log_interval']
    action_embedding_dim                    = config['action_embedding_dim']

    # State shape (channels, H, W) or (H, W)
    image_size                              = encoder_params.get('image_size', [10, 10])
    input_channels                          = encoder_params.get('input_channels', 1)
    if isinstance(image_size, int):
        state_shape                         = (input_channels, image_size, image_size)
    else:
        state_shape                         = (input_channels, image_size[0], image_size[1])

    dataset                                 = ReplayBufferDataset(
        buffer_path=buffer_path,
        num_color_selection_fns=num_color_selection_fns,
        num_selection_fns=num_selection_fns,
        num_transform_actions=num_transform_actions,
        num_arc_colors=11,
        state_shape=state_shape,
        mode='color_only'
    )
    # Split into train/val (80/20)
    val_size                                = int(0.2 * len(dataset))
    train_size                              = len(dataset) - val_size
    train_dataset, val_dataset              = random_split(dataset, [train_size, val_size])
    train_loader                            = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader                              = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    state_encoder                           = StateEncoder(
        image_size=image_size,
        input_channels=input_channels,
        latent_dim=latent_dim,
        encoder_params=encoder_params
    ).to(device)
    
    # Load pretrained encoder if specified
    if use_pretrained_encoder:
        if os.path.exists(pretrained_encoder_path):
            print(f"Loading pretrained encoder from {pretrained_encoder_path}")
            checkpoint = torch.load(pretrained_encoder_path, map_location=device)
            state_encoder.load_state_dict(checkpoint['state_encoder'])
            print("Pretrained encoder loaded successfully!")
            
            # Freeze encoder parameters if specified
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
    
    action_embedder                         = ActionEmbedder(
        num_actions=num_color_selection_fns, 
        embed_dim=action_embedding_dim, 
        dropout_p=0.1
        ).to(device)

    color_predictor                         = ColorPredictor(
        latent_dim,
        num_colors=11, 
        hidden_dim=color_predictor_hidden_dim, 
        action_embedding_dim=action_embedding_dim
        ).to(device)
    
    #color_predictor = TransformerColorPredictor(latent_dim, action_embedding_dim=action_embedding_dim, num_colors=11, transformer_depth=2, transformer_heads=4, transformer_dim_head=32, transformer_mlp_dim=512, transformer_dropout=0.3, mlp_hidden_dim=256).to(device)

    criterion                               = nn.CrossEntropyLoss()
    
    # Only include trainable parameters in the optimizer
    trainable_params = []
    if use_pretrained_encoder and freeze_pretrained_encoder:
        # If encoder is frozen, don't include its parameters
        trainable_params.extend(list(action_embedder.parameters()))
        trainable_params.extend(list(color_predictor.parameters()))
        print("Optimizer will only train action embedder and color predictor (encoder frozen)")
    
    else:
        # Include all parameters
        trainable_params.extend(list(state_encoder.parameters()))
        trainable_params.extend(list(action_embedder.parameters()))
        trainable_params.extend(list(color_predictor.parameters()))
        print("Optimizer will train all parameters")
    
    optimizer                               = optim.Adam(trainable_params, lr=learning_rate)

    type_stats = {
        "random": {"loss": 0.0, "correct": 0, "total": 0},
        "challenge": {"loss": 0.0, "correct": 0, "total": 0}
    }

    best_val_loss                           = float('inf')
    epochs_no_improve                       = 0
    patience                                = 50
    save_path                               = 'best_model_color_predictor.pth'
    wandb.init(project="color_predictor", config=config)
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print("-" * 80)
    
    # Main epoch progress bar
    epoch_pbar = tqdm(range(num_epochs), desc='Training Progress', ncols=120)
    
    for epoch in epoch_pbar:
        state_encoder.train()
        action_embedder.train()
        color_predictor.train()
        
        total_loss                          = 0
        
        # Create progress bar for training batches
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', 
                         leave=False, ncols=100)
        
        for i, batch in enumerate(train_pbar):
            state                           = batch['state'].to(device)  # (B, H, W) or (B, C, H, W)
            action_colour                   = batch['action_colour'].to(device)  # (B,)
            target_colour                   = batch['colour'].to(device)  # (B,)
            shape_w                         = batch['shape_w'].to(device)
            shape_h                         = batch['shape_h'].to(device)
            num_colors_grid                 = batch['num_colors_grid'].to(device)
            most_present_color              = batch['most_present_color'].to(device)
            least_present_color             = batch['least_present_color'].to(device)

            # State embedding
            if state.dim() == 3:
                # (B, H, W) -> (B, 1, H, W) for single channel
                state = state
            latent = state_encoder(
                state,
                shape_w=shape_w,
                shape_h=shape_h,
                num_unique_colors=num_colors_grid,
                most_common_color=most_present_color,
                least_common_color=least_present_color
            )  # (B, latent_dim)

            # Color selection one-hot
            action_colour_onehot            = one_hot(action_colour, num_color_selection_fns)  # (B, num_color_selection_fns)
            action_embedding                = action_embedder(action_colour_onehot)  # (B, 12)

            # Concatenate and predict
            logits                          = color_predictor(latent, action_embedding)  # (B, num_arc_colors)
            
            loss                            = criterion(logits, target_colour)
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            
            optimizer.step()

            # Log gradients to wandb
            for name, model in zip([
                'state_encoder', 'action_embedder', 'color_predictor'],
                [state_encoder, action_embedder, color_predictor]):
                for param_name, param in model.named_parameters():
                    if param.grad is not None:
                        wandb.log({f"gradients/{name}/{param_name}": wandb.Histogram(param.grad.cpu().data.numpy())}, step=epoch * len(train_loader) + i)

            total_loss                     += loss.item() * state.size(0)

            with torch.no_grad():
                preds = torch.argmax(logits, dim=1)
                # batch['transition_type'] is a list of strings, one per sample
                transition_types = batch['transition_type']
                for j, ttype in enumerate(transition_types):
                    if ttype in type_stats:
                        type_stats[ttype]["loss"] += loss.item()  # Optionally, you can use per-sample loss if you want more precision
                        type_stats[ttype]["correct"] += int(preds[j].item() == target_colour[j].item())
                        type_stats[ttype]["total"] += 1

            # Update progress bar with current loss
            train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})

        train_pbar.close()
        
        avg_loss = total_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate(color_predictor, state_encoder, action_embedder, val_loader, device, criterion, num_color_selection_fns)
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'Train Loss': f'{avg_loss:.4f}',
            'Val Loss': f'{val_loss:.4f}',
            'Val Acc': f'{val_acc:.4f}',
            'Best': f'{best_val_loss:.4f}'
        })
        
        # Check for improvement
        improvement_status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                'state_encoder': state_encoder.state_dict(),
                'action_embedder': action_embedder.state_dict(),
                'color_predictor': color_predictor.state_dict()
            }, save_path)
            improvement_status = f" ✓ New best model saved!"
        else:
            epochs_no_improve += 1
            improvement_status = f" ({epochs_no_improve} epochs without improvement)"
        
        # Print detailed epoch summary
        tqdm.write(f"Epoch {epoch+1:3d}/{num_epochs} | Train: {avg_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.4f}{improvement_status}")
        
        # Print type-specific statistics if available
        type_summary = []
        for ttype, stats in type_stats.items():
            if stats["total"] > 0:
                type_avg_loss = stats["loss"] / stats["total"]
                type_accuracy = stats["correct"] / stats["total"]
                type_summary.append(f"{ttype}: {type_accuracy:.3f} ({stats['total']})")
            stats["loss"] = 0.0  # Reset for next epoch
            stats["correct"] = 0
            stats["total"] = 0
        
        if type_summary:
            tqdm.write(f"         Type breakdown: {' | '.join(type_summary)}")
        
        # Early stopping check
        if epochs_no_improve >= patience:
            tqdm.write(f"\nEarly stopping at epoch {epoch+1} due to no improvement for {patience} epochs.")
            break

    epoch_pbar.close()
    
    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {save_path}")
    print("="*80)

    wandb.finish()

if __name__ == "__main__":
    train_color_predictor() 