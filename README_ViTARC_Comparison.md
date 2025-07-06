# ViTARC Positional Embedding Comparison Framework

This framework provides a comprehensive comparison of different positional embedding combinations for the enhanced ViTARC state encoder on the color predictor task.

## Overview

The framework systematically tests 16 different positional embedding configurations, including:

- **APE (Absolute Positional Embedding)**: Different types of absolute positional embeddings
- **RPE (Relative Positional Embedding)**: Alibi-based relative positional embeddings
- **OPE (Object Positional Embedding)**: Object-aware positional embeddings
- **Mixer Strategies**: Different ways to combine input and positional embeddings

## Files

- `train_ViTARC.py`: Main training script that runs all experiments
- `run_vitarc_comparison.py`: Utility script for running and analyzing experiments
- `README_ViTARC_Comparison.md`: This documentation file

## Experiment Configurations

The framework tests the following configurations:

### Baseline Configurations
1. **baseline_no_pe**: Basic 2D sinusoidal APE (baseline)
2. **baseline_learned_ape**: Standard learned absolute positional embeddings
3. **baseline_sinusoidal_ape**: Standard sinusoidal absolute positional embeddings

### 2D APE Configurations
4. **ape_2d_basic**: 2D sinusoidal APE with default mixing
5. **ape_2d_weighted_sum**: 2D sinusoidal APE with weighted sum mixing
6. **ape_2d_learnable_scaling**: 2D sinusoidal APE with learnable scaling
7. **ape_2d_layer_norm**: 2D sinusoidal APE with layer norm mixing

### RPE Configurations
8. **rpe_two_slope_alibi**: Two-slope Alibi RPE with 2D APE
9. **rpe_four_diag_slope_alibi**: Four-diagonal-slope Alibi RPE with 2D APE
10. **rpe_two_slope_no_abs**: Two-slope Alibi RPE without absolute distance

### OPE Configurations
11. **ope_basic**: Object positional embeddings with 2D APE
12. **ope_with_rpe**: Object positional embeddings with RPE and 2D APE
13. **ope_with_four_diag_rpe**: Object positional embeddings with Four-diag RPE

### Advanced Configurations
14. **positional_attention**: Positional attention mixing with RPE
15. **hardcoded_normalization**: Hardcoded normalization mixing with RPE

### Full ViTARC Configurations
16. **full_vitarc**: Full ViTARC: 2D APE + RPE + OPE
17. **full_vitarc_four_diag**: Full ViTARC with Four-diagonal RPE

## Usage

### Quick Start

```bash
# Run all experiments
python train_ViTARC.py

# Or use the utility script
python run_vitarc_comparison.py
```

### Advanced Usage

```bash
# Run experiments without Weights & Biases logging
python run_vitarc_comparison.py --no-wandb

# Only analyze existing results
python run_vitarc_comparison.py --action analyze

# Generate a report from existing results
python run_vitarc_comparison.py --action report

# Specify custom results directory
python run_vitarc_comparison.py --action analyze --results-dir my_experiments
```

### Configuration Requirements

Make sure your `config.yaml` file contains the necessary parameters:

```yaml
# Dataset
buffer_path: "path/to/your/replay_buffer.pkl"
batch_size: 32
num_workers: 4

# Model architecture
latent_dim: 256
encoder_params:
  image_size: [10, 10]  # or single int for square images
  input_channels: 1
  transformer_dim: 64
  depth: 4
  heads: 8
  mlp_dim: 512
  dropout: 0.2
  emb_dropout: 0.2
  colors_vocab_size: 11

# Training
num_epochs: 100
learning_rate: 1e-4
log_interval: 100

# Action embedders
action_embedders:
  action_color_embedder:
    num_actions: 13
    embed_dim: 12

# Color predictor
color_predictor:
  hidden_dim: 512

# Other parameters
num_arc_colors: 11
```

## Output

The framework generates several output files in the `vitarc_experiments/` directory:

### Results Files
- `results.json`: Detailed results in JSON format
- `results.csv`: Results in CSV format for easy analysis
- `experiment_report.md`: Comprehensive markdown report

### Model Checkpoints
- `best_model_{config_name}.pth`: Best model checkpoint for each configuration

### Analysis Output
The framework provides:
- **Performance ranking**: Configurations ranked by validation accuracy
- **Component analysis**: Performance breakdown by APE type, RPE type, OPE usage, and mixer strategy
- **Efficiency analysis**: Training time and parameter count comparisons
- **Best combinations**: Recommendations for optimal configurations

## Example Output

```
================================================================================
EXPERIMENT SUMMARY
================================================================================
Rank Configuration          Val Acc    Val Loss   Time (s)   Params    
--------------------------------------------------------------------------------
1    full_vitarc            0.8542     0.4123     145.2      2,401,234
2    ope_with_rpe           0.8498     0.4201     132.8      2,401,234
3    rpe_two_slope_alibi    0.8445     0.4356     128.1      2,356,890
4    ape_2d_weighted_sum    0.8401     0.4489     121.4      2,356,890
5    full_vitarc_four_diag  0.8389     0.4512     148.9      2,401,234

Top 3 configurations:
1. full_vitarc: 0.8542 accuracy
   Description: Full ViTARC: 2D APE + RPE + OPE
2. ope_with_rpe: 0.8498 accuracy
   Description: Object positional embeddings with RPE and 2D APE
3. rpe_two_slope_alibi: 0.8445 accuracy
   Description: Two-slope Alibi RPE with 2D APE
```

## Understanding the Results

### Key Metrics
- **Val Acc**: Validation accuracy (higher is better)
- **Val Loss**: Validation loss (lower is better)
- **Time (s)**: Training time in seconds
- **Params**: Number of trainable parameters

### Component Analysis
The framework analyzes performance by:
- **APE Type**: Which absolute positional embedding works best
- **RPE Type**: Impact of relative positional embeddings
- **OPE Usage**: Whether object positional embeddings help
- **Mixer Strategy**: Best way to combine embeddings

### Recommendations
The framework will identify:
1. **Best overall configuration**
2. **Most efficient configuration** (best performance/time ratio)
3. **Most parameter-efficient configuration**
4. **Best configuration for each component type**

## Customization

### Adding New Configurations

To add new configurations, modify the `get_experiment_configurations()` function in `train_ViTARC.py`:

```python
configs.append(
    PositionalEmbeddingConfig(
        name="my_custom_config",
        ape_type="SinusoidalAPE2D",
        rpe_type="Two-slope-Alibi",
        rpe_abs=True,
        use_OPE=True,
        ape_mixer_strategy="my_custom_strategy",
        description="My custom configuration description"
    )
)
```

### Modifying Training Parameters

You can modify training parameters by editing the `train_single_configuration()` function:

```python
# Change early stopping patience
patience = 30  # Default is 20

# Change maximum epochs
num_epochs = min(base_config['num_epochs'], 50)  # Default is 100
```

## Performance Considerations

- **Training Time**: Each experiment takes approximately 5-20 minutes depending on your hardware
- **Memory Usage**: The framework requires sufficient GPU memory for the transformer models
- **Disk Space**: Model checkpoints and results require approximately 1-2GB of disk space

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in config.yaml
2. **Dataset Not Found**: Check buffer_path in config.yaml
3. **Wandb Authentication**: Set `--no-wandb` flag or configure wandb properly

### Tips for Faster Experimentation

1. **Reduce epochs**: Set lower max_epochs for quick testing
2. **Use subset**: Implement subset functionality for testing specific configurations
3. **Parallel runs**: Run different configurations on different GPUs if available

## Citation

If you use this framework in your research, please cite:

```bibtex
@misc{vitarc_comparison_2024,
  title={ViTARC Positional Embedding Comparison Framework},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/your-repo}}
}
```

## License

This framework is licensed under the same license as your main project. 