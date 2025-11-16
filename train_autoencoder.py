import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from src.models.state_encoder import StateEncoder
from src.models.state_decoder import StateDecoder
from src.data.replay_buffer_dataset import ReplayBufferDataset
import torch.nn.functional as F
import subprocess

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

def evaluate(encoder, decoder, dataloader, device):
    encoder.eval()
    decoder.eval()
    total_loss = 0
    total_grid_loss = 0
    total_shape_loss = 0
    total_color_stats_loss = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            state = batch['state'].to(device)
            shape_w = batch['shape_w'].to(device)
            shape_h = batch['shape_h'].to(device)
            num_colors_grid = batch['num_colors_grid'].to(device)
            most_present_color = batch['most_present_color'].to(device)
            least_present_color = batch['least_present_color'].to(device)

            latent = encoder(
                state,
                shape_w=shape_w,
                shape_h=shape_h,
                num_unique_colors=num_colors_grid,
                most_common_color=most_present_color,
                least_common_color=least_present_color
            )
            decoder_output = decoder(latent)
            loss, loss_dict = autoencoder_loss(
                decoder_output, state, shape_h, shape_w, most_present_color, least_present_color, num_colors_grid
            )
            total_loss += loss.item() * state.size(0)
            total_grid_loss += loss_dict['grid_loss'] * state.size(0)
            total_shape_loss += loss_dict['shape_loss'] * state.size(0)
            total_color_stats_loss += loss_dict['color_stats_loss'] * state.size(0)
            total += state.size(0)

    avg_loss = total_loss / total
    avg_grid_loss = total_grid_loss / total
    avg_shape_loss = total_shape_loss / total
    avg_color_stats_loss = total_color_stats_loss / total

    return avg_loss, {
        'grid_loss': avg_grid_loss,
        'shape_loss': avg_shape_loss,
        'color_stats_loss': avg_color_stats_loss
    }

def train_autoencoder(cfg: DictConfig):
    # --- WANDB INIT EARLY ---
    try:
        if not wandb.run:
            wandb.login()
    except Exception as e:
        print(f"wandb login failed or already logged in: {e}")
    # Log all config values to wandb at the very start
    wandb_config = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(project="autoencoder", config=wandb_config)

    buffer_path = cfg.data.autoencoder.buffer_path
    fast_buffer_path = buffer_path + '.fast.pt'
    if not os.path.exists(fast_buffer_path):
        print(f"Fast buffer {fast_buffer_path} not found. Preprocessing...")
        subprocess.run(['python', 'scripts/preprocess_buffer.py', buffer_path, fast_buffer_path], check=True)
    else:
        print(f"Using fast buffer: {fast_buffer_path}")
    latent_dim = cfg.latent_dim
    encoder_params = OmegaConf.to_container(cfg.model.encoder.encoder_params, resolve=True)
    decoder_params = OmegaConf.to_container(cfg.model.decoder.decoder_params, resolve=True)
    batch_size = cfg.training.autoencoder.batch_size
    num_epochs = cfg.training.autoencoder.num_epochs
    learning_rate = cfg.training.autoencoder.learning_rate
    num_workers = cfg.training.autoencoder.num_workers
    log_interval = cfg.training.autoencoder.log_interval

    image_size = encoder_params.get('image_size', [10, 10])
    input_channels = encoder_params.get('input_channels', 1)
    if isinstance(image_size, int):
        state_shape = (input_channels, image_size, image_size)
    else:
        state_shape = (input_channels, image_size[0], image_size[1])

    dataset = ReplayBufferDataset(
        buffer_path=fast_buffer_path,
        num_color_selection_fns=cfg.num_color_selection_fns,
        num_selection_fns=cfg.num_selection_fns,
        num_transform_actions=cfg.num_transform_actions,
        num_arc_colors=11,
        state_shape=state_shape,
        mode='full'
    )

    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # --- Load pretrained encoder if specified ---
    use_pretrained_encoder = OmegaConf.select(cfg, 'use_pretrained_encoder', default=False)
    pretrained_encoder_path = OmegaConf.select(cfg, 'pretrained_encoder_path', default='best_model_autoencoder.pth')
    freeze_pretrained_encoder = OmegaConf.select(cfg, 'freeze_pretrained_encoder', default=False)

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

    state_decoder = StateDecoder(
        image_size=image_size,
        latent_dim=latent_dim,
        decoder_params=decoder_params
    ).to(device)

    # Only include trainable parameters in the optimizer
    if use_pretrained_encoder and freeze_pretrained_encoder:
        optimizer = optim.AdamW(list(state_decoder.parameters()), lr=learning_rate)
    else:
        optimizer = optim.AdamW(list(state_encoder.parameters()) + list(state_decoder.parameters()), lr=learning_rate)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    patience = 50
    save_path = 'best_model_next_state_predictor.pth'

    try:
        for epoch in range(num_epochs):
            state_encoder.train()
            state_decoder.train()
            total_loss = 0
            total_grid_loss = 0
            total_shape_loss = 0
            total_color_stats_loss = 0
            for i, batch in enumerate(train_loader):
                state = batch['state'].to(device)
                shape_w = batch['shape_w'].to(device)
                shape_h = batch['shape_h'].to(device)
                num_colors_grid = batch['num_colors_grid'].to(device)
                most_present_color = batch['most_present_color'].to(device)
                least_present_color = batch['least_present_color'].to(device)

                latent = state_encoder(
                    state,
                    shape_w=shape_w,
                    shape_h=shape_h,
                    num_unique_colors=num_colors_grid,
                    most_common_color=most_present_color,
                    least_common_color=least_present_color
                )
                decoder_output = state_decoder(latent)

                loss, loss_dict = autoencoder_loss(
                    decoder_output, state, shape_h, shape_w, most_present_color, least_present_color, num_colors_grid
                )

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(state_encoder.parameters()) + list(state_decoder.parameters()),
                    max_norm=1.0
                )
                optimizer.step()

                total_loss += loss.item() * state.size(0)
                total_grid_loss += loss_dict['grid_loss'] * state.size(0)
                total_shape_loss += loss_dict['shape_loss'] * state.size(0)
                total_color_stats_loss += loss_dict['color_stats_loss'] * state.size(0)

                if (i + 1) % log_interval == 0:
                    print(f"\rEpoch {epoch+1} Batch {i+1}/{len(train_loader)} Loss: {loss.item():.4f}", end='', flush=True)
                    wandb.log({
                        "batch_loss": loss.item(),
                        "batch_grid_loss": loss_dict['grid_loss'],
                        "batch_shape_loss": loss_dict['shape_loss'],
                        "batch_color_stats_loss": loss_dict['color_stats_loss'],
                        "epoch": epoch + 1,
                        "batch": i + 1
                    })

            avg_loss = total_loss / len(train_dataset)
            avg_grid_loss = total_grid_loss / len(train_dataset)
            avg_shape_loss = total_shape_loss / len(train_dataset)
            avg_color_stats_loss = total_color_stats_loss / len(train_dataset)
            val_loss, val_loss_dict = evaluate(state_encoder, state_decoder, val_loader, device)
            print(f"\rEpoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}", end='', flush=True)
            wandb.log({
                "train_loss": avg_loss,
                "train_grid_loss": avg_grid_loss,
                "train_shape_loss": avg_shape_loss,
                "train_color_stats_loss": avg_color_stats_loss,
                "val_loss": val_loss,
                "val_grid_loss": val_loss_dict['grid_loss'],
                "val_shape_loss": val_loss_dict['shape_loss'],
                "val_color_stats_loss": val_loss_dict['color_stats_loss']
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                torch.save({
                    'state_encoder': state_encoder.state_dict(),
                    'state_decoder': state_decoder.state_dict(),
                }, save_path)
                print(f"\nNew best model saved to {save_path}")
            else:
                epochs_no_improve += 1
                print(f"\nNo improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} due to no improvement in validation loss for {patience} epochs.")
                break
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (KeyboardInterrupt). Exiting gracefully...")
    finally:
        wandb.finish()

if __name__ == "__main__":
    if not GlobalHydra().is_initialized():
        initialize(config_path="conf", version_base=None)
    cfg = compose(config_name="config_autoencoder")
    train_autoencoder(cfg)
