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
        # Encode input to latent
        latent = self.encoder(x)
        latent = latent.unsqueeze(0)  # (1, batch, latent_dim)
        output = self.decoder(latent, tgt_mask=tgt_mask)
        return output

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
    'latent_dim': 128,
    'n_attention_head': 4,
    'num_layers': 4
}

autoencoder = Autoencoder(encoder_params, decoder_params)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
autoencoder = autoencoder.to(device)

optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()  # Use a suitable loss for your data

num_epochs = 10

for epoch in range(num_epochs):
    autoencoder.train()
    total_loss = 0
    for batch in dataloader:
        x = batch[0].to(device)
        output = autoencoder(x)
        output = output.squeeze(0)
        loss = loss_fn(output, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader):.4f}")

torch.save(autoencoder.encoder.state_dict(), "encoder_weights.pth")
print("Encoder weights saved to encoder_weights.pth")

autoencoder.eval()  # Set to eval mode if not training
with torch.no_grad():
    output = autoencoder(state_tensor.float())  # If your model expects float, cast as needed





