import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import yaml
import pickle
from src.models.state_encoder import StateEncoder
from src.models.color_predictor import ColorPredictor
from src.data import ReplayBufferDataset
from torch.utils.data import Dataset
# --- Config Loader ---
def load_config(config_path="unified_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# --- One-hot encoding utility ---
def one_hot(indices, num_classes):
    return torch.nn.functional.one_hot(indices, num_classes=num_classes).float()

# --- Validation Metrics ---
def evaluate(model, encoder, dataloader, device, criterion, num_color_selection_fns):
    model.eval()
    encoder.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            state = batch['state'].to(device)
            action_colour = batch['action_colour'].to(device)
            target_colour = batch['colour'].to(device)
            if state.dim() == 3:
                state = state.unsqueeze(1)
            latent = encoder(state.float())
            action_colour_onehot = one_hot(action_colour, num_color_selection_fns)
            x = torch.cat([latent, action_colour_onehot], dim=1)
            logits = model(x)
            loss = criterion(logits, target_colour)
            total_loss += loss.item() * state.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == target_colour).sum().item()
            total += state.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
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
    config = load_config()
    buffer_path = config['buffer_path']
    #encoder_type = config['encoder_type']
    latent_dim = config['latent_dim']
    encoder_params = config['encoder_params'][encoder_type]
    num_color_selection_fns = config['num_color_selection_fns']
    num_selection_fns = config['num_selection_fns']
    num_transform_actions = config['num_transform_actions']
    #num_arc_colors = config['num_arc_colors']
    color_predictor_hidden_dim = config['color_predictor']['hidden_dim']
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
        mode='color_only'
    )
    # Split into train/val (80/20)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device('mps' if torch.cuda.is_available() else 'cpu')
    
    # Extract parameters for StateEncoder
    patch_size = encoder_params.get('patch_size', 2)  # Default for ViT
    
    state_encoder = StateEncoder(
        encoder_type=encoder_type,
        image_size=image_size,
        patch_size=patch_size,
        input_channels=input_channels,
        latent_dim=latent_dim,
        encoder_params=encoder_params
    ).to(device)
    color_predictor = ColorPredictor(latent_dim + num_color_selection_fns, num_arc_colors, color_predictor_hidden_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(state_encoder.parameters()) + list(color_predictor.parameters()), lr=learning_rate)

    for epoch in range(num_epochs):
        state_encoder.train()
        color_predictor.train()
        total_loss = 0
        for i, batch in enumerate(train_loader):
            state = batch['state'].to(device)  # (B, H, W) or (B, C, H, W)
            action_colour = batch['action_colour'].to(device)  # (B,)
            target_colour = batch['colour'].to(device)  # (B,)


            # State embedding
            if state.dim() == 3:
                # (B, H, W) -> (B, 1, H, W) for single channel
                state = state.unsqueeze(1)
            latent = state_encoder(state.float())  # (B, latent_dim)

            # Color selection one-hot
            action_colour_onehot = one_hot(action_colour, num_color_selection_fns)  # (B, num_color_selection_fns)

            # Concatenate and predict
            x = torch.cat([latent, action_colour_onehot], dim=1)  # (B, latent_dim + num_color_selection_fns)
            logits = color_predictor(x)  # (B, num_arc_colors)

            loss = criterion(logits, target_colour)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * state.size(0)

            if (i + 1) % log_interval == 0:
                print(f"Epoch {epoch+1} Batch {i+1}/{len(train_loader)} Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader.dataset)
        val_loss, val_acc = evaluate(color_predictor, state_encoder, val_loader, device, criterion, num_color_selection_fns)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

if __name__ == "__main__":
    train_color_predictor() 