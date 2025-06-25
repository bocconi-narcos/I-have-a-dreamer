# Extending the Project

This project is designed for easy extension and experimentation.

## Custom Buffers
- Implement buffer loading in `ReplayBufferDataset` to use your own data.

## New Encoders
- Add new encoder classes to `src/models/`.
- Update `StateEncoder` to support your new encoder.

## Advanced Losses
- Integrate self-supervised or auxiliary losses from `src/losses/` if desired.
- See `src/losses/` for examples (VICReg, Barlow Twins, DINO).

## Custom Training Loops
- Use or adapt scripts in `src/training_loops/` for more complex workflows.

## Utilities
- Use scripts in `src/utils/` for data collection, plotting, and more.

## New Sub-action Module
- To add a new sub-action module (e.g., selection mask prediction):
  - Implement a new predictor (e.g., SelectionMaskPredictor) and encoder (e.g., MaskEncoder) in `src/models/`.
  - Add a new training loop (e.g., `train_selection_mask_predictor`) and dataset class if needed.
  - Update the config YAML to include all relevant hyperparameters.
  - Update the documentation (`docs/model.md`, `docs/training.md`, `docs/buffer.md`) to describe the new module and its data flow. 