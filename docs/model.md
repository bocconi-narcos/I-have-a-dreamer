# Model Architecture

The model consists of five main components, all using modern Transformer-based architectures:

## 1. State Encoder
- Configurable as MLP, CNN, or ViT.
- Converts the grid state into a latent vector.
- Parameters are set in `config.yaml` under `encoder_params`.

## 2. Action Encoding
- The color selection index (`action['colour']`), selection index (`action['selection']`), and transform index (`action['transform']`) are one-hot encoded.
- The number of functions is set by `num_color_selection_fns`, `num_selection_fns`, and `num_transform_actions`.

## 3. Color Predictor
- An MLP that takes the concatenated state embedding and color action encoding.
- Outputs a probability distribution over possible colors (set by `num_arc_colors`).

## 4. Selection Mask Predictor (Transformer-based)
- Takes the encoded state, selection action one-hot, and color prediction as separate inputs.
- Projects inputs to the same dimension if needed.
- Stacks as sequence of length 2: `[encoded_state, concat(selection_action, color_pred)]`.
- Uses Transformer encoder with positional encodings.
- Outputs a latent mask representation.
- The ground truth selection mask is encoded by a Mask Encoder (configurable: MLP, CNN, ViT) to produce the target latent mask.
- Loss is MSE or optionally VICReg (configurable).

## 5. Next State Predictor (Transformer-based)
- Takes the encoded state, transform action one-hot, and predicted latent mask as separate inputs.
- Projects inputs to the same dimension if needed.
- Stacks as sequence of length 2: `[encoded_state, concat(transform_action, latent_mask)]`.
- Uses Transformer encoder with positional encodings.
- Outputs a predicted latent next state.
- Loss is MSE or optionally VICReg (configurable).

## 6. Reward Predictor (Transformer-based)
- Takes two latent vectors (`z_t`, `z_{t+1}`) as separate inputs.
- Projects to the same dimension if needed.
- Stacks as sequence of length 2, prepends CLS token, adds positional encodings.
- Uses Transformer encoder to model interactions.
- Outputs a scalar reward prediction.
- Uses MLP head on CLS token output.

## 7. Continuation Predictor (Transformer-based)
- Takes two latent vectors (`z_t`, `z_{t+1}`) as separate inputs.
- Projects to the same dimension if needed.
- Stacks as sequence of length 2, prepends CLS token, adds positional encodings.
- Uses Transformer encoder to model interactions.
- Outputs a probability of continuation (boolean).
- Uses MLP head + sigmoid on CLS token output.

## Data Flow
1. State → State Encoder → Latent Vector
2. Action['colour'] → One-hot Encoding
3. [Latent Vector, Color One-hot] → Concatenation → Color Predictor → Color Prediction
4. Action['selection'] → One-hot Encoding
5. [Latent Vector, Selection One-hot, Color Prediction] → Selection Mask Predictor (Transformer) → Predicted Latent Mask
6. Selection Mask → Mask Encoder → Target Latent Mask
7. Action['transform'] → One-hot Encoding
8. [Latent Vector, Transform One-hot, Latent Mask] → Next State Predictor (Transformer) → Predicted Next Latent
9. [Latent Vector, Predicted Next Latent] → Reward Predictor (Transformer) → Reward Prediction
10. [Latent Vector, Predicted Next Latent] → Continuation Predictor (Transformer) → Continuation Probability 