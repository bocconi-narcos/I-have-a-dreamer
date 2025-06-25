# Project Overview

**I-have-a-dreamer** is a modular framework for training a color predictor model on grid-based state transitions. It supports flexible state encoders (MLP, CNN, ViT), a transformer-based next state predictor, and a configurable training pipeline, making it easy to adapt to different data and research needs.

## Main Features
- Modular state encoder selection (MLP, CNN, ViT)
- Transformer-based next state (transformation) prediction
- Configurable via YAML file
- Simple, extensible training loop
- Dummy data generation for rapid prototyping
- Designed for research and experimentation

## High-Level Workflow
1. **Load buffer**: A list of transitions (see [Buffer/Data Structure](./buffer.md)).
2. **Configure model**: All settings in the relevant config YAML file.
3. **Train**: The model learns to predict the correct color and the next latent state given a state and action.
4. **Validate**: Reports loss and accuracy after each epoch for both color and transformation prediction.

For details on each step, see the relevant documentation pages. 