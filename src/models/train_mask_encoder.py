import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import wandb
from src.models.mask_encoder_new import MaskEncoder
from src.models.mask_decoder_new import MaskDecoder
from src.data.replay_buffer_dataset import ReplayBufferDataset

# --- WANDB SETUP ---
# If you haven't already, run 'wandb login' in your terminal to authenticate your account.

# --- Config ---
image_size = (10, 10)
vocab_size = 12  # Number of mask/object categories (should match your config)
latent_dim = 64
batch_size = 32
num_epochs = 100
learning_rate = 1e-3
num_workers = 0
padding_value = -1
log_interval = 10
patience = 10
save_path = 'best_model_mask_color_predictor.pth'

# --- Dataset ---
dataset = ReplayBufferDataset(
    buffer_path="data/buffer_100.pt",
    num_color_selection_fns=22,
    num_selection_fns=8,
    num_transform_actions=8,
    num_arc_colors=vocab_size,
    state_shape=image_size,
    mode='mask_only'
)
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# --- Model ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
encoder = MaskEncoder(
    image_size=image_size,
    vocab_size=vocab_size,
    emb_dim=latent_dim,
    depth=4,
    heads=8,
    mlp_dim=256,
    dropout=0.2,
    emb_dropout=0.2,
    padding_value=padding_value
).to(device)
decoder = MaskDecoder(
    image_size=image_size,
    latent_dim=latent_dim,
    decoder_params={'colors_vocab_size': vocab_size, 'transformer_dim': latent_dim},
    padding_value=padding_value
).to(device)
color_predictor = nn.Linear(latent_dim, vocab_size).to(device)

optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()) + list(color_predictor.parameters()), lr=learning_rate)
mask_loss_fn = nn.CrossEntropyLoss(ignore_index=padding_value)
color_loss_fn = nn.CrossEntropyLoss()

# --- WANDB INIT ---
wandb.init(project="mask_color_autoencoder", config={
    'image_size': image_size,
    'vocab_size': vocab_size,
    'latent_dim': latent_dim,
    'batch_size': batch_size,
    'num_epochs': num_epochs,
    'learning_rate': learning_rate,
    'num_workers': num_workers,
    'padding_value': padding_value,
    'buffer_path': "data/buffer_100.pt"
})

def evaluate(encoder, decoder, color_predictor, dataloader, device):
    encoder.eval()
    decoder.eval()
    color_predictor.eval()
    total_mask_loss = 0
    total_color_loss = 0
    total = 0
    with torch.no_grad():
        for batch in dataloader:
            mask = batch['state'].to(device)
            color_label = batch['most_present_color'].to(device)
            latent = encoder(mask)
            logits = decoder(latent).permute(0, 3, 1, 2)
            mask_loss = mask_loss_fn(logits, mask)
            color_logits = color_predictor(latent)
            color_loss = color_loss_fn(color_logits, color_label)
            total_mask_loss += mask_loss.item() * mask.size(0)
            total_color_loss += color_loss.item() * mask.size(0)
            total += mask.size(0)
    avg_mask_loss = total_mask_loss / total
    avg_color_loss = total_color_loss / total
    return avg_mask_loss, avg_color_loss

best_val_loss = float('inf')
epochs_no_improve = 0

for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    color_predictor.train()
    total_mask_loss = 0
    total_color_loss = 0
    for i, batch in enumerate(train_loader):
        mask = batch['state'].to(device)
        color_label = batch['most_present_color'].to(device)
        latent = encoder(mask)
        logits = decoder(latent).permute(0, 3, 1, 2)
        mask_loss = mask_loss_fn(logits, mask)
        color_logits = color_predictor(latent)
        color_loss = color_loss_fn(color_logits, color_label)
        loss = mask_loss + color_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_mask_loss += mask_loss.item() * mask.size(0)
        total_color_loss += color_loss.item() * mask.size(0)
        if (i + 1) % log_interval == 0:
            print(f"Epoch {epoch+1} Batch {i+1}/{len(train_loader)} Loss: {loss.item():.4f}")
    avg_mask_loss = total_mask_loss / len(train_loader.dataset)
    avg_color_loss = total_color_loss / len(train_loader.dataset)
    val_mask_loss, val_color_loss = evaluate(encoder, decoder, color_predictor, val_loader, device)
    print(f"Epoch {epoch+1}/{num_epochs} - Train Mask Loss: {avg_mask_loss:.4f} | Train Color Loss: {avg_color_loss:.4f} | Val Mask Loss: {val_mask_loss:.4f} | Val Color Loss: {val_color_loss:.4f}")
    wandb.log({
        "train_mask_loss": avg_mask_loss,
        "train_color_loss": avg_color_loss,
        "val_mask_loss": val_mask_loss,
        "val_color_loss": val_color_loss,
        "epoch": epoch+1
    })
    if val_mask_loss < best_val_loss:
        best_val_loss = val_mask_loss
        epochs_no_improve = 0
        torch.save({
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict(),
            'color_predictor': color_predictor.state_dict()
        }, save_path)
        print(f"New best model saved to {save_path}")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s)")
    if epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch+1} due to no improvement in validation loss for {patience} epochs.")
        break

wandb.finish()

# %%
