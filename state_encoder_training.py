import torch
import torch.nn as nn
from src.models.state_encoder import StateEncoder
from src.models.state_decoder import StateDecoder
# from src.models.state_decoder import StateDecoder
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import torch.optim as optim
import h5py

class Autoencoder(nn.Module):
    def __init__(self, encoder_params, decoder_params):
        super().__init__()
        self.encoder = StateEncoder(**encoder_params)
        self.decoder = StateDecoder(**decoder_params)

    def forward(self, x, tgt_mask=None):
        latent = self.encoder(x)
        shape_row_logits, shape_col_logits, grid_logits = self.decoder(latent, H=x.shape[2], W=x.shape[3], dropout_eval=False)
        return grid_logits  # (batch, N, vocab_size)

buffer_path = "data/buffer.h5"
with h5py.File(buffer_path, "r") as f:
    state_np = f["state"][:]  # shape: (num_samples, H, W) or (num_samples, 1, H, W)

print("State shape:", state_np.shape)
state_tensor = torch.from_numpy(state_np)

# If state_tensor is [num_samples, H, W], add a channel dimension
if state_tensor.ndim == 3:
    state_tensor = state_tensor.unsqueeze(1)  # [num_samples, 1, H, W]
num_samples = state_tensor.shape[0]

train_size = int(0.8 * num_samples)
val_size = num_samples - train_size

# Ensure both splits are at least 1
if train_size == 0:
    train_size = 1
    val_size = num_samples - 1
if val_size == 0:
    val_size = 1
    train_size = num_samples - 1

print(f"Splitting: train_size={train_size}, val_size={val_size}, total={num_samples}")

dataset = TensorDataset(state_tensor)
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

encoder_params = {
    'image_size': 10,
    'input_channels': 1,
    'latent_dim': 128,
    'encoder_params': {'depth': 4, 'heads': 4, 'mlp_dim': 256},               
}
decoder_params = {
    'emb_state_dim': 128,   # latent_dim from encoder
    'emb_dim': 128,         # transformer embedding dimension
    'num_layers': 4,
    'max_rows': 10,
    'max_cols': 10,
    'vocab_size': 11,       # number of color categories
    'mlp_dim': 256
}

autoencoder = Autoencoder(encoder_params, decoder_params)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
autoencoder = autoencoder.to(device)

optimizer = optim.AdamW(autoencoder.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

num_epochs = 30
best_val_loss = float('inf')
best_encoder_weights = None

for epoch in range(num_epochs):
    autoencoder.train()
    total_loss = 0
    for batch in train_loader:
        x = batch[0].to(device)
        output = autoencoder(x)
        target = x.squeeze(1).long().view(x.shape[0], -1)
        loss = loss_fn(output.permute(0, 2, 1), target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    # Validation
    autoencoder.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            x = batch[0].to(device)
            output = autoencoder(x)
            target = x.squeeze(1).long().view(x.shape[0], -1)
            loss = loss_fn(output.permute(0, 2, 1), target)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # Save best encoder weights based on lowest avg val loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_encoder_weights = autoencoder.encoder.state_dict()
        print(f"New best model found at epoch {epoch+1} with avg val loss {best_val_loss:.4f}")

# After training, save the best encoder weights
if best_encoder_weights is not None:
    torch.save(best_encoder_weights, "encoder_weights_best.pth")
    print(f"Best encoder weights saved to encoder_weights_best.pth with val loss {best_val_loss:.4f}")

autoencoder.eval()  # Set to eval mode if not training
with torch.no_grad():
    state_tensor = state_tensor.to(device)
    output = autoencoder(state_tensor.float())  # If your model expects float, cast as needed

# Example: Run prediction on a batch from the validation set
autoencoder.eval()
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
autoencoder = autoencoder.to(device)

with torch.no_grad():
    for batch in val_loader:
        x = batch[0].to(device)
        output = autoencoder(x)  # This will run encoder and decoder
        # output shape depends on your model's forward
        print("Predicted output shape:", output.shape)
        break  # Just run for one batch as an example





