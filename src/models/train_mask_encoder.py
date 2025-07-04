import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.models.mask_encoder_new import MaskEncoder
from src.models.mask_decoder_new import MaskDecoder  # You need to implement or import this
from src.data.replay_buffer_dataset import ReplayBufferDataset

# --- Config ---
image_size = (10, 10)
vocab_size = 12  # Number of mask/object categories (should match your config)
latent_dim = 64
batch_size = 32
num_epochs = 50
learning_rate = 1e-3
num_workers = 0
padding_value = -1

# --- Dataset ---
dataset = ReplayBufferDataset(
    buffer_path="data/buffer.h5",
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

optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss(ignore_index=padding_value)

# --- Training Loop ---
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    total_loss = 0
    for batch in train_loader:
        mask = batch['state'].to(device)  # (B, H, W)
        latent = encoder(mask)            # (B, latent_dim)
        logits = decoder(latent)          # (B, H, W, vocab_size)
        logits = logits.permute(0, 3, 1, 2)  # (B, vocab_size, H, W)
        target = mask  # (B, H, W)
        loss = loss_fn(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * mask.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.4f}")

    # Optionally: validation loop here

# Optionally: save model
torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, 'best_model_mask_autoencoder.pth')
    
# %%
