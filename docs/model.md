# Model Architecture

The model consists of three main components:

## 1. State Encoder
- Configurable as MLP, CNN, or ViT.
- Converts the grid state into a latent vector.
- Parameters are set in `color_predictor_config.yaml` under `encoder_params`.

## 2. Action Encoding
- The color selection index (`action['colour']`) is one-hot encoded.
- The number of color selection functions is set by `num_color_selection_fns`.

## 3. Color Predictor
- An MLP that takes the concatenated state embedding and action encoding.
- Outputs a probability distribution over possible colors (set by `num_arc_colors`).

## Data Flow
1. State → State Encoder → Latent Vector
2. Action['colour'] → One-hot Encoding
3. [Latent Vector, One-hot] → Concatenation → Color Predictor → Output 