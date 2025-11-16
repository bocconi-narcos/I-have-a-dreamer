# Configuration Guide

This project uses **Hydra** for configuration management, providing modular, composable, and overrideable configuration files.

## Overview

All model choices, hyperparameters, and data paths are configured through Hydra's modular configuration system located in the `conf/` directory.

### Benefits of Hydra Configuration

- **Modular**: Related configs grouped logically
- **Overrideable**: Easy parameter tuning via command-line
- **Type-safe**: OmegaConf provides validation
- **Composable**: Easy to combine different configs
- **Maintainable**: Clear separation of concerns

## Configuration Structure

```
conf/
├── config.yaml                    # Main config (for most training scripts)
├── config_autoencoder.yaml         # Autoencoder-specific config
├── data/
│   ├── default.yaml               # Main data configuration
│   └── autoencoder/
│       └── default.yaml           # Autoencoder data config
├── model/
│   ├── encoder/
│   │   ├── vit.yaml              # ViT encoder parameters
│   │   └── autoencoder.yaml      # Autoencoder encoder parameters
│   └── predictors/
│       └── default.yaml           # All predictor configurations
└── training/
    ├── default.yaml               # Main training hyperparameters
    └── autoencoder/
        └── default.yaml          # Autoencoder training parameters
```

## Main Configuration (`config.yaml`)

The main configuration file is used by most training scripts (`train_color_predictor.py`, `train_selection_predictor.py`, etc.).

### Key Sections

#### Data Configuration (`conf/data/default.yaml`)
```yaml
buffer_path: "data/buffer_all_solved_100.pt"
```

#### Model Configuration

**Encoder** (`conf/model/encoder/vit.yaml`):
```yaml
encoder_params:
  image_size: [10, 10]
  input_channels: 1
  depth: 4
  heads: 16
  mlp_dim: 512
  transformer_dim: 96
  dropout: 0.1
  emb_dropout: 0.1
  scaled_position_embeddings: true
  colors_vocab_size: 12
```

**Predictors** (`conf/model/predictors/default.yaml`):
- Color Predictor
- Selection Mask Predictor
- Next State Predictor
- Reward Predictor
- Continuation Predictor
- Action Embedders

#### Training Configuration (`conf/training/default.yaml`)
```yaml
batch_size: 96
num_epochs: 100
learning_rate: 0.0001
num_workers: 0
log_interval: 10
```

## Autoencoder Configuration (`config_autoencoder.yaml`)

Used by `train_autoencoder.py` and `train_step_distance_encoder.py`.

### Structure
- **Data**: `conf/data/autoencoder/default.yaml`
- **Encoder**: `conf/model/encoder/autoencoder.yaml`
- **Decoder**: `conf/model/decoder/autoencoder.yaml`
- **Training**: `conf/training/autoencoder/default.yaml`

## Usage

### Basic Usage

Run training scripts with default configuration:

```bash
# Main config scripts
python train_color_predictor.py
python train_selection_predictor.py
python train_next_state_predictor.py
python train_reward_predictor.py
python train_full_model.py
python train_step_distance_mlp.py

# Autoencoder config scripts
python train_autoencoder.py
python train_step_distance_encoder.py
```

### Command-Line Overrides

Override any configuration value from the command line:

```bash
# Override batch size
python train_color_predictor.py training.batch_size=128

# Override learning rate
python train_color_predictor.py training.learning_rate=0.001

# Override multiple values
python train_color_predictor.py training.batch_size=64 training.learning_rate=0.0005

# Override nested values
python train_color_predictor.py model.encoder.encoder_params.depth=6

# Override data path
python train_color_predictor.py data.buffer_path=data/my_buffer.pt

# Override autoencoder config
python train_autoencoder.py training.autoencoder.batch_size=512
```

### Common Overrides

#### Training Hyperparameters
```bash
# Batch size
python train_color_predictor.py training.batch_size=128

# Learning rate
python train_color_predictor.py training.learning_rate=0.001

# Number of epochs
python train_color_predictor.py training.num_epochs=200

# Log interval
python train_color_predictor.py training.log_interval=20
```

#### Model Architecture
```bash
# Encoder depth
python train_color_predictor.py model.encoder.encoder_params.depth=6

# Encoder heads
python train_color_predictor.py model.encoder.encoder_params.heads=32

# Latent dimension
python train_color_predictor.py latent_dim=512
```

#### Data Configuration
```bash
# Buffer path
python train_color_predictor.py data.buffer_path=data/my_buffer.pt

# Number of workers
python train_color_predictor.py training.num_workers=4
```

## Configuration Access in Code

### Loading Configuration

All training scripts follow this pattern:

```python
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

def train_function(cfg: DictConfig):
    # Access config values
    buffer_path = cfg.data.buffer_path
    batch_size = cfg.training.batch_size
    latent_dim = cfg.latent_dim
    
    # Convert nested configs to dicts (for compatibility)
    encoder_params = OmegaConf.to_container(
        cfg.model.encoder.encoder_params, 
        resolve=True
    )
    
    # Access optional values with defaults
    use_pretrained = OmegaConf.select(
        cfg, 
        'use_pretrained_encoder', 
        default=False
    )

if __name__ == "__main__":
    if not GlobalHydra().is_initialized():
        initialize(config_path="conf", version_base=None)
    cfg = compose(config_name="config")
    train_function(cfg)
```

### Config Access Patterns

**Direct access:**
```python
batch_size = cfg.training.batch_size
latent_dim = cfg.latent_dim
buffer_path = cfg.data.buffer_path
```

**Nested access:**
```python
num_actions = cfg.model.predictors.action_embedders.action_color_embedder.num_actions
hidden_dim = cfg.model.predictors.color_predictor.hidden_dim
```

**Convert to dict (for compatibility):**
```python
encoder_params = OmegaConf.to_container(
    cfg.model.encoder.encoder_params, 
    resolve=True
)
# Now use as dict: encoder_params['depth']
```

**Optional values with defaults:**
```python
use_pretrained = OmegaConf.select(
    cfg, 
    'use_pretrained_encoder', 
    default=False
)
```

## Configuration Files Reference

### Main Config Files

#### `conf/config.yaml`
Main configuration file that composes all other configs. Contains:
- Training switches (`use_ground_truth`, `use_decoder_loss`)
- Encoder type and settings
- Model saving paths
- SOTA training parameters

#### `conf/config_autoencoder.yaml`
Autoencoder-specific configuration. Composes:
- Autoencoder data config
- Autoencoder encoder config
- Decoder config
- Autoencoder training config

### Data Configs

#### `conf/data/default.yaml`
- `buffer_path`: Path to replay buffer file

#### `conf/data/autoencoder/default.yaml`
- `buffer_path`: Path to replay buffer for autoencoder training

### Model Configs

#### `conf/model/encoder/vit.yaml`
ViT encoder parameters:
- `image_size`: Grid dimensions
- `input_channels`: Number of input channels
- `depth`: Number of transformer layers
- `heads`: Number of attention heads
- `mlp_dim`: MLP dimension
- `transformer_dim`: Transformer dimension
- `dropout`: Dropout rate
- `emb_dropout`: Embedding dropout
- `scaled_position_embeddings`: Use scaled positional embeddings
- `colors_vocab_size`: Color vocabulary size

#### `conf/model/predictors/default.yaml`
Contains configurations for all predictors:
- Color Predictor
- Selection Mask Predictor
- Next State Predictor
- Reward Predictor
- Continuation Predictor
- Action Embedders (color, selection, transform)

### Training Configs

#### `conf/training/default.yaml`
Main training hyperparameters:
- `batch_size`: Batch size for training
- `num_epochs`: Number of training epochs
- `learning_rate`: Learning rate
- `num_workers`: DataLoader workers
- `log_interval`: Logging interval

#### `conf/training/autoencoder/default.yaml`
Autoencoder-specific training parameters:
- Same structure as main training config
- Separate values for autoencoder training

## Advanced Usage

### Multiple Overrides

Override multiple values at once:

```bash
python train_color_predictor.py \
    training.batch_size=128 \
    training.learning_rate=0.001 \
    model.encoder.encoder_params.depth=6 \
    latent_dim=512
```

### Config Composition

Hydra automatically composes configs based on the `defaults` list in the main config file. You can override which configs are used:

```bash
# Use different encoder config (if available)
python train_color_predictor.py model/encoder=cnn
```

### Saving Overrides

Create a new config file with your overrides:

```bash
# This creates a new config file with your overrides
python train_color_predictor.py training.batch_size=128 --config-path=conf --config-name=config
```

## Migration from Old Config System

If you're migrating from the old YAML config system:

### Old Way
```python
import yaml

def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

config = load_config()
batch_size = config['batch_size']
encoder_params = config['encoder_params']
```

### New Way (Hydra)
```python
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

if not GlobalHydra().is_initialized():
    initialize(config_path="conf", version_base=None)
cfg = compose(config_name="config")

batch_size = cfg.training.batch_size
encoder_params = OmegaConf.to_container(
    cfg.model.encoder.encoder_params, 
    resolve=True
)
```

## Tips

1. **Use command-line overrides** for quick experiments without editing files
2. **Check config structure** by printing `cfg` or using `OmegaConf.to_yaml(cfg)`
3. **Validate configs** - Hydra will catch invalid config paths and types
4. **Organize related configs** in the modular structure for easy maintenance
5. **Use defaults** in config files for common values

## Troubleshooting

### Config Not Found
If you get a "config not found" error:
- Check that `config_path="conf"` is correct relative to your script
- Verify the config file exists in `conf/` directory
- Ensure config name matches (e.g., `config` or `config_autoencoder`)

### Override Not Working
- Use dot notation: `training.batch_size=128` not `training['batch_size']=128`
- Check the exact path in the config structure
- Verify the value type matches (string vs int vs float)

### Import Errors
- Ensure `hydra-core` is installed: `pip install hydra-core`
- Check that Hydra is initialized before using `compose()`

## See Also

- [Hydra Documentation](https://hydra.cc/)
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/)
- [Project Overview](./overview.md)
- [Model Architecture](./model.md)
- [Training Guide](./training.md)
