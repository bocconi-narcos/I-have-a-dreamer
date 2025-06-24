# Configuration Guide

All model choices, hyperparameters, and data paths are set in `color_predictor_config.yaml`.

## Example Configuration
```yaml
buffer_path: "buffer.pkl"  # Path to replay buffer file (use dummy data if not found)
encoder_type: "mlp"        # Options: "mlp", "cnn", "vit"
latent_dim: 128
encoder_params:
  mlp:
    num_hidden_layers: 2
    hidden_dim: 256
    activation_fn_str: "relu"
    dropout_rate: 0.1
    input_channels: 1
    image_size: [10, 10]
  cnn:
    ...
  vit:
    ...
num_color_selection_fns: 19
color_predictor_hidden_dim: 128
num_arc_colors: 9
batch_size: 32
num_epochs: 10
learning_rate: 0.001
num_workers: 1
log_interval: 10
```

## How to Change Settings
- **Model type**: Set `encoder_type` to `mlp`, `cnn`, or `vit`.
- **Hyperparameters**: Adjust `latent_dim`, `batch_size`, `learning_rate`, etc.
- **Encoder details**: Edit the relevant section under `encoder_params`.
- **Data**: Change `buffer_path` to point to your buffer file.

See comments in the YAML file for more details. 