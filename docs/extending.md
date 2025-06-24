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