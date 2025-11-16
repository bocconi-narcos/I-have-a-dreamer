# Hydra Migration Timeline - Remaining Tasks

## Overview
Complete migration of remaining config files and training scripts to Hydra.

## Timeline

### Phase 1: Config Migration (30 min)
**Goal**: Migrate `config_autoencoder.yaml` to Hydra structure

1. **Create Autoencoder Config Structure** (10 min)
   - Create `conf/config_autoencoder.yaml` (main config)
   - Create `conf/data/autoencoder.yaml` (autoencoder-specific data config)
   - Create `conf/model/encoder/autoencoder.yaml` (autoencoder encoder params)
   - Create `conf/model/decoder/autoencoder.yaml` (autoencoder decoder params)
   - Create `conf/training/autoencoder.yaml` (autoencoder training params)

2. **Verify Config Structure** (5 min)
   - Test config loading
   - Verify all values match original

### Phase 2: Script Migration - Autoencoder Scripts (45 min)
**Goal**: Migrate scripts that use `config_autoencoder.yaml`

3. **Migrate `train_autoencoder.py`** (20 min)
   - Replace `load_config()` with Hydra
   - Update config access patterns
   - Test script loads correctly

4. **Migrate `train_step_distance_encoder.py`** (15 min)
   - Replace `load_config()` with Hydra
   - Update config access patterns
   - Test script loads correctly

5. **Test Autoencoder Configs** (10 min)
   - Verify both scripts work
   - Test command-line overrides

### Phase 3: Script Migration - Main Training Scripts (2 hours)
**Goal**: Migrate scripts that use `config.yaml`

6. **Migrate `train_selection_predictor.py`** (25 min)
   - Replace `load_config()` with Hydra
   - Update config access patterns
   - Test script loads correctly

7. **Migrate `train_next_state_predictor.py`** (25 min)
   - Replace `load_config()` with Hydra
   - Update config access patterns
   - Test script loads correctly

8. **Migrate `train_reward_predictor.py`** (25 min)
   - Replace `load_config()` with Hydra
   - Update config access patterns
   - Test script loads correctly

9. **Migrate `train_full_model.py`** (30 min)
   - Replace `load_config()` with Hydra
   - Update config access patterns
   - Test script loads correctly

10. **Migrate `train_step_distance_mlp.py`** (15 min)
    - Replace `load_config()` with Hydra
    - Update config access patterns
    - Test script loads correctly

### Phase 4: Testing & Cleanup (30 min)
**Goal**: Ensure everything works and clean up

11. **Comprehensive Testing** (20 min)
    - Test all migrated scripts load configs correctly
    - Test command-line overrides for each script
    - Verify no regressions

12. **Documentation & Cleanup** (10 min)
    - Update migration documentation
    - Create summary of changes
    - Mark old config files as deprecated (optional)

## Total Estimated Time: ~3.5 hours

## Files to Migrate

### Config Files:
- ✅ `config.yaml` → Already migrated
- ⏳ `config_autoencoder.yaml` → To migrate

### Training Scripts:
- ✅ `train_color_predictor.py` → Already migrated
- ⏳ `train_autoencoder.py` → To migrate
- ⏳ `train_selection_predictor.py` → To migrate
- ⏳ `train_next_state_predictor.py` → To migrate
- ⏳ `train_reward_predictor.py` → To migrate
- ⏳ `train_full_model.py` → To migrate
- ⏳ `train_step_distance_encoder.py` → To migrate
- ⏳ `train_step_distance_mlp.py` → To migrate

### Analysis Scripts (Lower Priority):
- ⏳ `inspect_states.py` → Optional
- ⏳ `state_analysis/distance_regression_train.py` → Optional
- ⏳ `state_analysis/pca_state_encoder_analysis.py` → Optional
- ⏳ `distance_analysis/state_distance_analysis.py` → Optional

## Success Criteria

- ✅ All config files migrated to Hydra structure
- ✅ All training scripts use Hydra configs
- ✅ Command-line overrides work for all scripts
- ✅ No regressions in functionality
- ✅ Clean, organized config structure
- ✅ Documentation updated

