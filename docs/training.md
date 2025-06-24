# Training Loop

The training loop processes the buffer in batches and updates the model to predict the correct color.

## Steps Per Batch
1. **Color Selection Encoding**: Extract `action['colour']` and one-hot encode it.
2. **State Embedding**: Pass `state` through the chosen state encoder.
3. **Color Prediction**: Concatenate the state embedding and action encoding, and pass through the color predictor MLP.
4. **Loss and Backpropagation**: Compute cross-entropy loss with the true color (`colour`), and backpropagate through both the color predictor and the state encoder.

## Validation
- After each epoch, the model is evaluated on a validation split.
- Reports loss and accuracy.

## Customization
- All hyperparameters and model choices are set in `color_predictor_config.yaml`. 