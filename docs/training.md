# Training Loop

The training loop processes the buffer in batches and updates the model to predict both the correct color and the next latent state.

## Steps Per Batch
1. **Color Selection Encoding**: Extract `action['colour']` and one-hot encode it.
2. **Transform Action Encoding**: Extract `action['transform']` and one-hot encode it.
3. **Selection Action Encoding**: Extract `action['selection']` and one-hot encode it.
4. **Selection Mask Prediction**: Concatenate the state embedding and selection action encoding, and pass through the selection mask predictor to produce a predicted latent mask.
5. **Selection Mask Target**: Pass the ground truth selection mask through the mask encoder to produce the target latent mask.
6. **Color Prediction**: Concatenate the state embedding and color action encoding, and pass through the color predictor MLP.
7. **Transformation Prediction**: Concatenate the state embedding, transform action encoding, and (optionally) a latent mask, and pass through the next state predictor (transformer).
8. **Loss and Backpropagation**:
   - Compute cross-entropy loss with the true color (`colour`), and backpropagate through both the color predictor and the state encoder.
   - Compute MSE (or VICReg, if enabled) loss between predicted and true next latent state, and backpropagate through the next state predictor and state encoder.
   - Compute MSE (or VICReg, if enabled) loss between predicted and target latent mask, and backpropagate through the selection mask predictor and mask encoder.

## Validation
- After each epoch, the model is evaluated on a validation split.
- Reports loss and accuracy for both color and transformation prediction.

## Customization
- All hyperparameters and model choices are set in the relevant config YAML file. 