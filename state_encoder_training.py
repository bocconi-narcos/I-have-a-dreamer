import torch
import torch.nn as nn
from src.models.state_encoder import StateEncoder
from src.models.state_decoder import StateDecoder
# from src.models.state_decoder import StateDecoder
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self, encoder_params, decoder_params):
        super().__init__()
        self.encoder = StateEncoder(**encoder_params)
        self.decoder = StateDecoder(**decoder_params)

    def forward(self, x, tgt_mask=None):
        latent = self.encoder(x)
        shape_row_logits, shape_col_logits, grid_logits = self.decoder(latent, H=x.shape[2], W=x.shape[3], dropout_eval=False)
        # grid_logits: (batch, N, vocab_size)
        return grid_logits

state = np.array([
    [ 7,  0,  7,  7,  8,  7, -1, -1, -1, -1],
    [ 7,  7,  7,  7,  7,  7, -1, -1, -1, -1],
    [ 7,  7,  7,  7,  7,  7, -1, -1, -1, -1],
    [ 7,  7,  7,  7,  7,  7, -1, -1, -1, -1],
    [ 7,  8,  7,  7,  0,  7, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
], dtype=np.int8)

# Convert to torch tensor and add batch and channel dimensions
state_tensor = torch.from_numpy(state).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 10, 10)

num_samples = 1000
image_size = 10
input_channels = 1
x_data = torch.randn(num_samples, input_channels, image_size, image_size)
dataset = TensorDataset(x_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

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

optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

num_epochs = 20
best_loss = float('inf')
best_encoder_weights = None

for epoch in range(num_epochs):
    autoencoder.train()
    total_loss = 0
    for batch in dataloader:
        x = batch[0].to(device)
        output = autoencoder(x)
        target = x.squeeze(1).long().view(x.shape[0], -1)  # (batch, N)
        loss = loss_fn(output.permute(0, 2, 1), target)  # CrossEntropyLoss expects (batch, vocab, N)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_encoder_weights = autoencoder.encoder.state_dict()
        print(f"New best model found at epoch {epoch+1} with loss {best_loss:.4f}")

if best_encoder_weights is not None:
    torch.save(best_encoder_weights, "encoder_weights_best.pth")
    print(f"Best encoder weights saved to encoder_weights_best.pth with loss {best_loss:.4f}")

autoencoder.eval()  # Set to eval mode if not training
with torch.no_grad():
    state_tensor = state_tensor.to(device)
    output = autoencoder(state_tensor.float())  # If your model expects float, cast as needed





