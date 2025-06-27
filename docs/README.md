# I-have-a-dreamer Documentation

Welcome to the documentation for the **I-have-a-dreamer** project.

This documentation provides an overview of the project, guides for usage, configuration, and details on the model and data pipeline.

**Now supports Transformer-based architecture for all predictors:**
- **Reward Predictor**: Transformer-based reward prediction from latent states
- **Continuation Predictor**: Transformer-based continuation probability prediction
- **Selection Mask Predictor**: Transformer-based latent mask prediction
- **Next State Predictor**: Transformer-based next state prediction
- **Color Predictor**: MLP-based color prediction

## Contents
- [Project Overview](./overview.md)
- [Buffer/Data Structure](./buffer.md)
- [Configuration Guide](./configuration.md)
- [Model Architecture](./model.md)
- [Training Loop](./training.md)
- [Extending the Project](./extending.md)
- [Project Structure](./structure.md)
- [License](../LICENSE)

For the quickest start, see the [Project Overview](./overview.md). 