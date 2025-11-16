# Hydra Migration Plan

## Current State Analysis

### Current Configuration System
- **Files**: `config.yaml`, `config_autoencoder.yaml`
- **Loading**: Simple `yaml.safe_load()` pattern
- **Access**: Dictionary access (`config['key']`)
- **Scripts**: Multiple training scripts load configs independently

### Files Using Config
1. `train_color_predictor.py` - uses `config.yaml`
2. `train_selection_predictor.py` - uses `config.yaml`
3. `train_next_state_predictor.py` - uses `config.yaml`
4. `train_reward_predictor.py` - uses `config.yaml`
5. `train_full_model.py` - uses `config.yaml`
6. `train_autoencoder.py` - uses `config_autoencoder.yaml`
7. `train_step_distance_encoder.py` - uses `config_autoencoder.yaml`
8. `train_step_distance_mlp.py` - uses `config.yaml`
9. Analysis scripts in `state_analysis/` - use `config.yaml`

## Hydra Benefits

1. **Configuration Composition**: Modular configs that can be combined
2. **Command-line Overrides**: Easy parameter overrides without editing files
3. **Config Groups**: Organize related configs (e.g., `model/`, `training/`, `data/`)
4. **Automatic Logging**: Configs automatically saved with runs
5. **Type Safety**: Better validation and type checking
6. **Multi-run**: Easy hyperparameter sweeps

## Migration Strategy

### Phase 1: Setup & Structure
1. Install `hydra-core`
2. Create `conf/` directory structure
3. Design modular config organization

### Phase 2: Config Migration
1. Split `config.yaml` into modular configs:
   - `conf/config.yaml` - main config
   - `conf/data/default.yaml` - data configs
   - `conf/model/encoder.yaml` - encoder configs
   - `conf/model/predictors/` - predictor configs
   - `conf/training/default.yaml` - training hyperparameters
   - `conf/training/color_predictor.yaml` - color predictor specific
   - etc.

2. Migrate `config_autoencoder.yaml` similarly

### Phase 3: Script Migration
1. Start with `train_color_predictor.py` as example
2. Update to use `@hydra.main` decorator
3. Replace `load_config()` with `cfg` parameter
4. Replace dictionary access with dot notation (`cfg.batch_size`)
5. Test thoroughly

### Phase 4: Full Migration
1. Migrate all remaining training scripts
2. Update analysis scripts
3. Test all scripts
4. Create migration guide

## Proposed Config Structure

```
conf/
├── config.yaml                    # Main config (defaults)
├── config_autoencoder.yaml        # Autoencoder main config
│
├── data/
│   ├── default.yaml              # Default data config
│   └── buffer_all_solved.yaml     # Specific buffer configs
│
├── model/
│   ├── encoder/
│   │   ├── vit.yaml              # ViT encoder config
│   │   ├── cnn.yaml               # CNN encoder config
│   │   └── mlp.yaml               # MLP encoder config
│   │
│   └── predictors/
│       ├── color_predictor.yaml
│       ├── selection_predictor.yaml
│       ├── reward_predictor.yaml
│       └── next_state_predictor.yaml
│
└── training/
    ├── default.yaml               # Default training params
    ├── color_predictor.yaml       # Color predictor specific
    └── autoencoder.yaml            # Autoencoder training
```

## Implementation Details

### Config Access Pattern Change

**Before**:
```python
config = load_config()
batch_size = config['batch_size']
encoder_params = config['encoder_params']
```

**After**:
```python
@hydra.main(version_base=None, config_path="conf", config_name="config")
def train(cfg):
    batch_size = cfg.training.batch_size
    encoder_params = cfg.model.encoder
```

### Command-line Overrides

**Before**: Edit config file
**After**: 
```bash
python train_color_predictor.py training.batch_size=128 model.encoder.depth=6
```

### Backward Compatibility

- Keep old config files temporarily
- Create adapter if needed
- Document migration path

## Testing Strategy

1. **Unit Tests**: Test config loading
2. **Integration Tests**: Test one training script end-to-end
3. **Regression Tests**: Verify same behavior as before
4. **Override Tests**: Test command-line overrides
5. **Multi-run Tests**: Test config composition

## Risks & Mitigation

1. **Risk**: Breaking existing workflows
   - **Mitigation**: Test thoroughly, keep old configs as backup

2. **Risk**: Learning curve for team
   - **Mitigation**: Create clear documentation and examples

3. **Risk**: Config complexity
   - **Mitigation**: Start simple, add complexity gradually

## Success Criteria

- ✅ All training scripts work with Hydra
- ✅ Command-line overrides work correctly
- ✅ Configs are modular and maintainable
- ✅ No regression in functionality
- ✅ Documentation updated
- ✅ Tests pass

