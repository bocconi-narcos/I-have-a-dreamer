import torch

# Load the checkpoint
checkpoint = torch.load('best_model_next_state_predictor.pth', map_location='cpu')

# View the keys in the checkpoint (these are usually the model state_dicts and other info)
print("Checkpoint keys:", checkpoint.keys())

# For example, to see the state dict for the state encoder:
if 'state_encoder' in checkpoint:
    print("State encoder keys:", checkpoint['state_encoder'].keys())

# To see all available model parts:
for key in checkpoint:
    print(f"{key}: type={type(checkpoint[key])}")