# Distance Regression Training Script

# - Step 1: Load config, encoder, and train dataset
# - Step 2: Define and initialize the MLP (128x128)
# - Step 3: Encode states and target states (with stop gradient)
# - Step 4: Train the MLP to predict distance
# - Step 5: Evaluate and report results

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
import numpy as np
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import wandb

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from src.models.state_encoder import StateEncoder
from train_step_distance_encoder import StepDistanceDataset
from src.models.predictors.reward_predictor import RewardPredictor

class DistanceMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=-1)
        return self.net(x).squeeze(-1)

# ---- Step 1: Load config, encoder, and train dataset ----
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_state_encoder_from_checkpoint(checkpoint_path, config):
    encoder_params = config['encoder_params']
    latent_dim = config['latent_dim']
    image_size = encoder_params.get('image_size', [10, 10])
    input_channels = encoder_params.get('input_channels', 1)
    encoder = StateEncoder(
        image_size=image_size,
        input_channels=input_channels,
        latent_dim=latent_dim,
        encoder_params=encoder_params
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'state_encoder' in checkpoint:
        encoder.load_state_dict(checkpoint['state_encoder'])
    else:
        raise KeyError("No 'state_encoder' key found in checkpoint")
    return encoder

CONFIG_PATH = "config.yaml"
CHECKPOINT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'weights', 'best_model_next_state_predictor.pth'))
BUFFER_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'buffer_500.pt'))

# Load config, encoder, and dataset
def main():
    print("Loading config...")
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.yaml'))
    config = load_config(config_path)
    print("Loading state encoder from checkpoint...")
    encoder = load_state_encoder_from_checkpoint(CHECKPOINT_PATH, config)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False  # Freeze encoder
    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    encoder.to(device)
    print(f"Using device: {device}")
    # Load buffer as dataset
    encoder_params = config['encoder_params']
    image_size = encoder_params.get('image_size', [10, 10])
    input_channels = encoder_params.get('input_channels', 1)
    if isinstance(image_size, int):
        state_shape = (input_channels, image_size, image_size)
    else:
        state_shape = (input_channels, image_size[0], image_size[1])
    dataset = StepDistanceDataset(
        buffer_path=BUFFER_PATH,
        num_color_selection_fns=config['action_embedders']['action_color_embedder']['num_actions'],
        num_selection_fns=config['action_embedders']['action_selection_embedder']['num_actions'],
        num_transform_actions=config['action_embedders']['action_transform_embedder']['num_actions'],
        num_arc_colors=config['num_arc_colors'],
        state_shape=state_shape,
        mode='full'
    )
    print(f"Loaded dataset with {len(dataset)} samples")
    # Split into train/val
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Prepare DataLoaders
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    def encode_batch(batch, encoder, device):
        state = batch['state'].to(device)
        target_state = batch['target_state'].to(device)
        shape_h = batch['shape_h'].to(device)
        shape_w = batch['shape_w'].to(device)
        num_colors_grid = batch['num_colors_grid'].to(device)
        most_present_color = batch['most_present_color'].to(device)
        least_present_color = batch['least_present_color'].to(device)
        with torch.no_grad():
            latent = encoder(
                state.to(torch.long),
                shape_h=shape_h,
                shape_w=shape_w,
                num_unique_colors=num_colors_grid,
                most_common_color=most_present_color,
                least_common_color=least_present_color
            )
            latent_target = encoder(
                target_state.to(torch.long),
                shape_h=shape_h,
                shape_w=shape_w,
                num_unique_colors=num_colors_grid,
                most_common_color=most_present_color,
                least_common_color=least_present_color
            )
        return latent, latent_target, batch['step_distance_to_target'].to(device)

    # ---- Step 2: Define and initialize the MLP (128x128) ----
    # class DistanceMLP(nn.Module):
    #     def __init__(self, input_dim, hidden_dim=128, dropout_p=0.2):
    #         super().__init__()
    #         self.net = nn.Sequential(
    #             nn.Linear(input_dim * 2, hidden_dim),
    #             nn.ReLU(),
    #             nn.Dropout(dropout_p),
    #             nn.Linear(hidden_dim, hidden_dim),
    #             nn.ReLU(),
    #             nn.Dropout(dropout_p),
    #             nn.Linear(hidden_dim, 1)
    #         )
    #     def forward(self, x1, x2):
    #         x = torch.cat([x1, x2], dim=-1)
    #         return self.net(x).squeeze(-1)

    # After train/val split
    latent_dim = config['latent_dim']
    mlp = DistanceMLP(input_dim=latent_dim, dropout_p=0.2).to(device)
    print(f"[DistanceMLP] Number of parameters: {sum(p.numel() for p in mlp.parameters())}")

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

    # ---- Step 4: Train the MLP to predict distance ----
    num_epochs = 20
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        list(mlp.parameters()) + 
        list(reward_predictor.parameters()), 
        lr=1e-3
    )
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        mlp.train()
        train_loss = 0
        total_reward_loss = 0
        for i, batch in enumerate(train_loader):
            latent, latent_target, true_distance = encode_batch(batch, encoder, device)
            pred_distance = mlp(latent, latent_target)
            reward = batch['reward'].to(device)
            pred_reward = reward_predictor(latent, latent_target)
            reward_loss = nn.MSELoss()(pred_reward.squeeze(-1), reward.float())
            loss = criterion(pred_distance, true_distance.float())
            total_loss = loss + reward_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            train_loss += loss.item() * latent.size(0)
            total_reward_loss += reward_loss.item() * latent.size(0)
            wandb.log({
                "batch_color_loss": loss.item(),
                "batch_reward_loss": reward_loss.item(),
                "batch_total_loss": total_loss.item(),
                "epoch": epoch + 1,
                "batch": i + 1
            })
        train_loss /= len(train_loader.dataset)
        # Validation
        mlp.eval()
        val_loss = 0
        total_reward_loss = 0
        preds = []
        trues = []
        with torch.no_grad():
            for batch in val_loader:
                latent, latent_target, true_distance = encode_batch(batch, encoder, device)
                pred_distance = mlp(latent, latent_target)
                reward = batch['reward'].to(device)
                pred_reward = reward_predictor(latent, latent_target)
                reward_loss = nn.MSELoss()(pred_reward.squeeze(-1), reward.float())
                loss = criterion(pred_distance, true_distance.float())
                total_loss = loss + reward_loss
                val_loss += loss.item() * latent.size(0)
                total_reward_loss += reward_loss.item() * latent.size(0)
                preds.append(pred_distance.cpu().numpy())
                trues.append(true_distance.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the full model (not just state_dict)
            torch.save({
                'state_encoder': encoder.state_dict(),
                'color_predictor': mlp.state_dict(),
                'mask_encoder': reward_predictor.state_dict(),
                'selection_mask_predictor': None,
                'next_state_predictor': None,
                'reward_predictor': reward_predictor.state_dict(),
                'colour_selection_embedder': None,
                'selection_embedder': None,
                'target_encoder': None
            }, 'best_distance_mlp_full.pth')
            print("  New best model saved (full model).")
    # ---- Step 5: Evaluate and report results ----
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    plt.figure(figsize=(6,6))
    plt.scatter(trues, preds, s=5, alpha=0.5)
    plt.xlabel('True Distance')
    plt.ylabel('Predicted Distance')
    plt.title('Distance Regression: True vs. Predicted')
    plt.plot([trues.min(), trues.max()], [trues.min(), trues.max()], 'r--')
    plt.tight_layout()
    plt.savefig('distance_regression_scatter.png', dpi=200)
    plt.show()
    print("Final best validation loss:", best_val_loss)
    print("Final total reward loss:", total_reward_loss)

if __name__ == "__main__":
    main() 