# Hydra Migration Guide

## Overview

This repository has been migrated from simple YAML config files to Hydra for better configuration management.

## What Changed

### Before (Old System)
- Single `config.yaml` file
- Simple `yaml.safe_load()` pattern
- Dictionary access: `config['key']`
- No command-line overrides

### After (Hydra System)
- Modular config structure in `conf/` directory
- Hydra-based loading with `@hydra.main` or `compose()`
- Dot notation access: `cfg.key.subkey`
- Command-line overrides: `python script.py training.batch_size=128`

## Config Structure

```
conf/
├── config.yaml                    # Main config (defaults)
├── data/
│   └── default.yaml              # Data configuration
├── model/
│   ├── encoder/
│   │   └── vit.yaml              # Encoder configuration
│   └── predictors/
│       └── default.yaml           # Predictor configurations
└── training/
    └── default.yaml               # Training hyperparameters
```

## Usage

### Basic Usage

```python
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# Initialize Hydra
if not GlobalHydra().is_initialized():
    initialize(config_path="conf", version_base=None)

# Compose config
cfg = compose(config_name="config")

# Access config values
batch_size = cfg.training.batch_size
learning_rate = cfg.training.learning_rate
latent_dim = cfg.latent_dim
```

### Command-Line Overrides

```bash
# Override batch size
python train_color_predictor.py training.batch_size=128

# Override multiple values
python train_color_predictor.py training.batch_size=128 training.learning_rate=0.001

# Override nested values
python train_color_predictor.py model.encoder.encoder_params.depth=6
```

### Accessing Config Values

**Old way**:
```python
config = load_config()
batch_size = config['batch_size']
encoder_params = config['encoder_params']
```

**New way**:
```python
cfg = compose(config_name="config")
batch_size = cfg.training.batch_size
encoder_params = OmegaConf.to_container(cfg.model.encoder.encoder_params, resolve=True)
```

## Migration Checklist

- [x] Install `hydra-core>=1.3.0`
- [x] Create `conf/` directory structure
- [x] Migrate `config.yaml` to modular configs
- [x] Update `train_color_predictor.py` to use Hydra
- [ ] Migrate remaining training scripts
- [ ] Update analysis scripts
- [ ] Test all scripts
- [ ] Update documentation

## Benefits

1. **Modular Configs**: Related configs grouped together
2. **Command-Line Overrides**: Easy experimentation without editing files
3. **Config Composition**: Combine different configs easily
4. **Type Safety**: Better validation with OmegaConf
5. **Automatic Logging**: Configs saved with runs

## Backward Compatibility

Old config files (`config.yaml`, `config_autoencoder.yaml`) are kept for reference but are no longer used by migrated scripts.

## Testing

Run tests to verify config loading:
```bash
pytest tests/test_hydra_config.py -v
```

## Troubleshooting

### Config not found
- Ensure `conf/` directory exists
- Check config path in `initialize(config_path="conf")`

### Attribute errors
- Use dot notation: `cfg.training.batch_size` not `cfg['training']['batch_size']`
- For optional values, use `OmegaConf.select(cfg, 'path', default=value)`

### Command-line overrides not working
- Ensure Hydra is initialized before composing config
- Use dot notation for nested paths: `training.batch_size=128`

