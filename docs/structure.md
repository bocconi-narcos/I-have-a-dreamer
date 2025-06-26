# Project Structure

```
I-have-a-dreamer/
  src/
    models/           # Model definitions (MLP, CNN, ViT, etc.)
    losses/           # Self-supervised/auxiliary loss functions (not used by default)
    utils/            # Utility functions (data, plotting, etc.)
    training_loops/   # Advanced training loops (not used by default)
    ...
  train_color_predictor.py   # Main training script
  color_predictor_config.yaml
  README.md
```

- **src/models/**: All model components, including encoders and predictors.
- **src/losses/**: Optional self-supervised/auxiliary loss functions.
- **src/utils/**: Data utilities, plotting, and helpers.
- **src/training_loops/**: Advanced/custom training loop scripts.
- **train_color_predictor.py**: Main entry point for training.
- **color_predictor_config.yaml**: All configuration and hyperparameters.
- **README.md**: Project summary and quickstart.

**Note:** Reward and continuation predictors now take both the encoded state and predicted next state as input, as a sequence (for transformer) or concatenation (for MLP). 