# Model Architecture

The model consists of four main components:

## 1. State Encoder
- Configurable as MLP, CNN, or ViT.
- Converts the grid state into a latent vector.
- Parameters are set in `color_predictor_config.yaml` under `encoder_params`.

## 2. Action Encoding
- The color selection index (`action['colour']`) and transform index (`action['transform']`) are one-hot encoded.
- The number of color selection and transform functions is set by `num_color_selection_fns` and `num_transform_fns`.

## 3. Selection Mask Prediction (NEW)
- Takes the latent state and one-hot selection action as input.
- Predicts a latent mask representation using an MLP or transformer block.
- The ground truth selection mask is encoded by a Mask Encoder (configurable: MLP, CNN, ViT) to produce the target latent mask.
- Loss is MSE or optionally VICReg (configurable).

## 4. Color Predictor
- An MLP that takes the concatenated state embedding and color action encoding.
- Outputs a probability distribution over possible colors (set by `num_arc_colors`).

## 5. Next State Predictor (Transformation Prediction Module)
- A transformer-based module that takes the current latent state, one-hot transform action, and (optionally) a latent mask.
- Outputs a predicted latent next state.
- Loss is MSE or optionally VICReg (configurable).

## Data Flow
1. State → State Encoder → Latent Vector
2. Action['colour'] → One-hot Encoding
3. [Latent Vector, Color One-hot] → Concatenation → Color Predictor → Output
4. Action['selection'] → One-hot Encoding
5. [Latent Vector, Selection One-hot] → Concatenation → Selection Mask Predictor → Predicted Latent Mask
6. Selection Mask → Mask Encoder → Target Latent Mask
7. Action['transform'] → One-hot Encoding
8. [Latent Vector, Transform One-hot, (Latent Mask)] → Concatenation → Next State Predictor (Transformer) → Predicted Next Latent 