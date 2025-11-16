# Project Structure

```
I-have-a-dreamer/
  conf/               # Hydra configuration files
    config.yaml       # Main config
    config_autoencoder.yaml  # Autoencoder config
    data/             # Data configurations
    model/            # Model configurations
    training/         # Training configurations
  src/
    models/           # Model definitions (MLP, CNN, ViT, etc.)
    losses/           # Self-supervised/auxiliary loss functions (not used by default)
    utils/            # Utility functions (data, plotting, etc.)
    training_loops/   # Advanced training loops (not used by default)
    ...
  train_*.py          # Training scripts (all use Hydra)
  README.md
```

- **conf/**: Hydra configuration files - modular, composable configs for all components.
- **src/models/**: All model components, including encoders and predictors.
- **src/losses/**: Optional self-supervised/auxiliary loss functions.
- **src/utils/**: Data utilities, plotting, and helpers.
- **src/training_loops/**: Advanced/custom training loop scripts.
- **train_*.py**: Training scripts - all use Hydra for configuration.
- **README.md**: Project summary and quickstart.

**Note:** Reward and continuation predictors now take both the encoded state and predicted next state as input, as a sequence (for transformer) or concatenation (for MLP). 