# I-have-a-dreamer: Visual Architecture Map

## System Overview Map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          REPLAY BUFFER DATASET                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Transition: {state, next_state, target_state, actions, masks, ...} │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STATE ENCODER (ViT)                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Input: Grid (H×W) + Metadata                                        │   │
│  │                                                                       │   │
│  │ 1. Color Embedding → Grid Tokens                                    │   │
│  │ 2. Positional Encoding (row + col)                                  │   │
│  │ 3. Special Tokens: [CLS, shape_h, shape_w, most_color,             │   │
│  │                     least_color, unique_count]                        │   │
│  │ 4. Transformer Blocks (Pre-norm, Self-Attention)                    │   │
│  │ 5. CLS Token Pooling                                                │   │
│  │                                                                       │   │
│  │ Output: z_t (B, latent_dim)                                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
        ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
        │ ACTION EMBEDDERS │ │ ACTION EMBEDDERS │ │ ACTION EMBEDDERS │
        │                  │ │                  │ │                  │
        │ Color:           │ │ Selection:       │ │ Transform:       │
        │ 23 → 32 dim      │ │ 6 → 32 dim        │ │ 18 → 32 dim      │
        └────────┬─────────┘ └────────┬─────────┘ └────────┬─────────┘
                 │                    │                    │
                 │                    │                    │
                 ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COLOR PREDICTOR (MLP)                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Input: [z_t, action_color_emb] → (B, latent_dim + 32)             │   │
│  │                                                                       │   │
│  │ MLP: Linear → ReLU → Linear                                          │   │
│  │                                                                       │   │
│  │ Output: color_logits (B, 12)                                         │   │
│  │         color_dist = softmax(color_logits)                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     │ color_dist
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SELECTION MASK PREDICTOR (Transformer)                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Input:                                                               │   │
│  │   - z_t (B, state_dim)                                              │   │
│  │   - action_selection_emb (B, 32)                                    │   │
│  │   - color_dist (B, 12)                                              │   │
│  │                                                                       │   │
│  │ 1. Project inputs to latent_mask_dim                                │   │
│  │ 2. Stack: [z_t_proj, concat(action_sel, color)]                     │   │
│  │ 3. Add positional encoding                                          │   │
│  │ 4. Transformer (depth=2, heads=2)                                    │   │
│  │ 5. Output projection                                                │   │
│  │                                                                       │   │
│  │ Output: pred_mask_latent (B, latent_mask_dim)                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     │ pred_mask_latent
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      NEXT STATE PREDICTOR (Transformer)                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Input:                                                               │   │
│  │   - z_t (B, state_dim)                                              │   │
│  │   - action_transform_onehot (B, 18)                                │   │
│  │   - pred_mask_latent (B, latent_mask_dim)                           │   │
│  │                                                                       │   │
│  │ 1. Project inputs to latent_dim                                     │   │
│  │ 2. Stack: [z_t_proj, concat(transform, mask)]                       │   │
│  │ 3. Add positional encoding                                          │   │
│  │ 4. Transformer (depth=2, heads=2)                                   │   │
│  │ 5. Use first token as output                                        │   │
│  │                                                                       │   │
│  │ Output: pred_z_{t+1} (B, latent_dim)                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
                    ▼                ▼                ▼
        ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
        │   REWARD         │ │  CONTINUATION    │ │   (Optional)     │
        │   PREDICTOR      │ │  PREDICTOR       │ │   DECODER        │
        │                  │ │                  │ │   LOSSES          │
        │ MLP:             │ │ MLP:             │ │                   │
        │ [z_t, z_{t+1},  │ │ [z_t, z_{t+1}]   │ │ State Decoder:   │
        │  z_target]      │ │ → sigmoid        │ │ z_{t+1} → grid   │
        │ → reward        │ │ → continue_prob  │ │                   │
        └──────────────────┘ └──────────────────┘ └───────────────────┘
```

---

## Detailed Data Flow

### Forward Pass Sequence

```
┌─────────────┐
│   BATCH     │
│  Loading    │
└──────┬──────┘
       │
       │ state, actions, targets, metadata
       ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: STATE ENCODING                                      │
│                                                              │
│  state (B,H,W) ──┐                                          │
│  shape_h (B) ────┤                                          │
│  shape_w (B) ────┤──→ StateEncoder ─→ z_t (B, latent_dim)  │
│  num_colors ─────┤                                          │
│  most_color ─────┤                                          │
│  least_color ────┘                                          │
└─────────────────────────────────────────────────────────────┘
       │
       │ z_t
       ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: ACTION EMBEDDING                                    │
│                                                              │
│  action_colour (B) ──→ ActionEmbedder ─→ emb_color (B,32)  │
│  action_selection (B) ─→ ActionEmbedder ─→ emb_sel (B,32)   │
│  action_transform (B) ─→ one_hot ─→ onehot_trans (B,18)    │
└─────────────────────────────────────────────────────────────┘
       │
       │ z_t, emb_color
       ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: COLOR PREDICTION                                    │
│                                                              │
│  [z_t, emb_color] ─→ ColorPredictor ─→ color_logits (B,12) │
│                                                              │
│  color_logits ─→ softmax ─→ color_dist (B,12)               │
└─────────────────────────────────────────────────────────────┘
       │
       │ z_t, emb_sel, color_dist
       ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: SELECTION MASK PREDICTION                          │
│                                                              │
│  [z_t, emb_sel, color_dist] ─→ SelectionMaskPredictor       │
│                                                              │
│  ┌──────────────────────────────────────────────┐          │
│  │ 1. Project to latent_mask_dim                │          │
│  │ 2. Stack as sequence [state, action+color]  │          │
│  │ 3. Transformer with positional encoding       │          │
│  │ 4. Output: pred_mask_latent (B, 256)         │          │
│  └──────────────────────────────────────────────┘          │
│                                                              │
│  Ground Truth:                                              │
│  selection_mask (B,H,W) ─→ MaskEncoder ─→ target_mask_latent│
│                                                              │
│  Loss: MSE(pred_mask_latent, target_mask_latent)          │
└─────────────────────────────────────────────────────────────┘
       │
       │ z_t, onehot_trans, pred_mask_latent
       ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: NEXT STATE PREDICTION                              │
│                                                              │
│  [z_t, onehot_trans, pred_mask_latent] ─→ NextStatePredictor│
│                                                              │
│  ┌──────────────────────────────────────────────┐          │
│  │ 1. Project to latent_dim                     │          │
│  │ 2. Stack as sequence [state, transform+mask] │          │
│  │ 3. Transformer with positional encoding       │          │
│  │ 4. Output: pred_z_{t+1} (B, latent_dim)     │          │
│  └──────────────────────────────────────────────┘          │
│                                                              │
│  Ground Truth:                                              │
│  next_state ─→ StateEncoder ─→ z_{t+1}                      │
│                                                              │
│  Loss: MSE(pred_z_{t+1}, z_{t+1})                           │
└─────────────────────────────────────────────────────────────┘
       │
       │ z_t, pred_z_{t+1}, z_target
       ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: REWARD PREDICTION                                  │
│                                                              │
│  [z_t, pred_z_{t+1}, z_target] ─→ RewardPredictor          │
│                                                              │
│  ┌──────────────────────────────────────────────┐          │
│  │ MLP Layers:                                  │          │
│  │   Linear(3*latent_dim → hidden_dim)          │          │
│  │   ReLU + Dropout                             │          │
│  │   Linear(hidden_dim → hidden_dim)           │          │
│  │   ReLU + Dropout                             │          │
│  │   Linear(hidden_dim → 1)                    │          │
│  │                                               │          │
│  │ Output: pred_reward (B, 1)                   │          │
│  └──────────────────────────────────────────────┘          │
│                                                              │
│  Loss: MSE(pred_reward, reward)                             │
└─────────────────────────────────────────────────────────────┘
       │
       │ z_t, pred_z_{t+1}
       ▼
┌─────────────────────────────────────────────────────────────┐
│ STEP 7: CONTINUATION PREDICTION                            │
│                                                              │
│  [z_t, pred_z_{t+1}] ─→ ContinuationPredictor              │
│                                                              │
│  ┌──────────────────────────────────────────────┐          │
│  │ Similar MLP structure                        │          │
│  │ Output: continuation_prob (B, 1)             │          │
│  │ Apply sigmoid                                │          │
│  └──────────────────────────────────────────────┘          │
│                                                              │
│  Loss: BCE(continuation_prob, 1 - done)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Module Dependency Graph

```
                    ┌─────────────────┐
                    │  Autoencoder    │
                    │  (Pre-training) │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ State Encoder   │
                    │   (Shared)      │
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Color      │   │   Selection   │   │   Next State  │
│  Predictor   │   │   Predictor   │   │   Predictor   │
│              │   │                │   │               │
│ Independent  │   │ Depends on:   │   │ Depends on:   │
│              │   │  - Color       │   │  - Selection  │
│              │   │                │   │               │
└──────┬───────┘   └───────┬────────┘   └───────┬───────┘
       │                   │                    │
       │                   │                    │
       └───────────────────┼────────────────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │   Reward &          │
                │   Continuation      │
                │   Predictors        │
                │                      │
                │ Depends on:          │
                │  - Next State        │
                └──────────────────────┘
```

---

## Training Strategy Map

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PROGRESSION                         │
└─────────────────────────────────────────────────────────────────┘

Phase 1: AUTOENCODER PRE-TRAINING
┌─────────────────────────────────────────────────────────────┐
│ train_autoencoder.py                                         │
│                                                              │
│  StateEncoder + StateDecoder                                │
│  Loss: Reconstruction (grid + shape + stats)                │
│  Output: Pre-trained encoder weights                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
Phase 2: COLOR PREDICTION
┌─────────────────────────────────────────────────────────────┐
│ train_color_predictor.py                                    │
│                                                              │
│  StateEncoder (frozen or fine-tuned) +                      │
│  ColorPredictor                                             │
│  Loss: Cross-Entropy(color_pred, color_target)            │
│  Mode: color_only                                           │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
Phase 3: SELECTION MASK PREDICTION
┌─────────────────────────────────────────────────────────────┐
│ train_selection_predictor.py                               │
│                                                              │
│  StateEncoder + ColorPredictor +                           │
│  MaskEncoder + SelectionMaskPredictor                      │
│  Loss: MSE/VICReg(mask_pred, mask_target) +                │
│        Cross-Entropy(color_pred, color_target)             │
│  Mode: selection_color                                     │
│  Options:                                                   │
│    - use_ground_truth: Use GT color vs predicted           │
│    - use_decoder_loss: Decoder loss vs latent loss         │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
Phase 4: NEXT STATE PREDICTION
┌─────────────────────────────────────────────────────────────┐
│ train_next_state_predictor.py                              │
│                                                              │
│  All previous modules + NextStatePredictor                 │
│  Loss: MSE/VICReg(z_{t+1}_pred, z_{t+1}_target) +          │
│        All previous losses                                  │
│  Mode: end_to_end                                          │
│  Features:                                                  │
│    - EMA target encoder for stability                      │
│    - Ground truth vs predicted inputs                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
Phase 5: REWARD & CONTINUATION
┌─────────────────────────────────────────────────────────────┐
│ train_reward_predictor.py                                   │
│ train_full_model.py (optional)                             │
│                                                              │
│  All modules                                                │
│  Loss: MSE(reward_pred, reward_target) +                   │
│        BCE(continue_pred, 1-done) +                         │
│        All previous losses                                  │
│  Mode: end_to_end                                          │
└─────────────────────────────────────────────────────────────┘
```

---

## Loss Function Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    LOSS COMPUTATION                          │
└─────────────────────────────────────────────────────────────┘

Color Loss:
  color_logits (B, 12) ──┐
                         ├─→ CrossEntropyLoss
  color_target (B) ───────┘
                         │
                         ▼
                    color_loss

Selection Mask Loss:
  pred_mask_latent (B, 256) ──┐
                               ├─→ MSE or VICReg
  target_mask_latent (B, 256) ─┘
                               │
                               ▼
                        selection_loss

Next State Loss:
  pred_z_{t+1} (B, latent_dim) ──┐
                                  ├─→ MSE or VICReg
  z_{t+1} (B, latent_dim) ───────┘
                                  │
                                  ▼
                           next_state_loss

Reward Loss:
  pred_reward (B, 1) ──┐
                       ├─→ MSELoss
  reward (B) ──────────┘
                       │
                       ▼
                  reward_loss

Continuation Loss:
  continuation_prob (B, 1) ──┐
                              ├─→ BCELoss
  (1 - done) (B) ─────────────┘
                              │
                              ▼
                      continuation_loss

Total Loss:
  total_loss = color_loss + 
               selection_loss + 
               next_state_loss + 
               reward_loss + 
               continuation_loss
```

---

## Configuration Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    CONFIG.YAML                               │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Data Configuration                                  │    │
│  │  - buffer_path                                     │    │
│  │  - batch_size                                      │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Model Architecture                                  │    │
│  │  - encoder_type: "vit"                             │    │
│  │  - latent_dim: 256                                  │    │
│  │  - encoder_params: {...}                            │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Module Configurations                               │    │
│  │  - color_predictor: {...}                           │    │
│  │  - selection_mask: {...}                            │    │
│  │  - next_state: {...}                                │    │
│  │  - reward_predictor: {...}                           │    │
│  └────────────────────────────────────────────────────┘    │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Training Parameters                                │    │
│  │  - learning_rate: 0.0003                            │    │
│  │  - num_epochs: 200                                  │    │
│  │  - use_ground_truth: false                         │    │
│  │  - use_decoder_loss: true                           │    │
│  └────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              TRAINING SCRIPTS                                │
│                                                              │
│  Each script loads config.yaml and:                         │
│  1. Initializes models with config parameters               │
│  2. Creates dataset with specified buffer_path              │
│  3. Sets up optimizers with learning_rate                  │
│  4. Trains for num_epochs                                  │
│  5. Saves best model to weights/                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

### 1. Why Transformer Architecture?
- **Self-attention**: Captures long-range dependencies in grid
- **Positional encoding**: Preserves spatial relationships
- **Pre-norm**: Better training stability
- **Modular**: Easy to stack layers

### 2. Why Latent Space Predictions?
- **Efficiency**: Predict in low-dimensional space
- **Abstraction**: Learn meaningful representations
- **Flexibility**: Can use decoder for reconstruction if needed

### 3. Why Progressive Training?
- **Stability**: Each module learns simpler task first
- **Dependencies**: Later modules depend on earlier ones
- **Debugging**: Easier to identify issues

### 4. Why Action Embeddings?
- **Learnability**: Model learns action semantics
- **Efficiency**: Dense vectors vs sparse one-hot
- **Generalization**: Similar actions have similar embeddings

### 5. Why Multiple Loss Options?
- **VICReg**: Better latent space structure
- **Decoder Loss**: Better reconstruction quality
- **MSE**: Simpler, faster training

---

## Summary Visualization

```
INPUT: Grid State + Actions
    │
    ▼
[State Encoder] → Latent Representation
    │
    ├─→ [Color Predictor] → Color
    │       │
    │       ▼
    └─→ [Selection Predictor] → Mask Latent
            │
            ▼
        [Next State Predictor] → Next State Latent
            │
            ├─→ [Reward Predictor] → Reward
            └─→ [Continuation Predictor] → Continue?
```

This modular architecture allows the system to learn complex sequential transformations by breaking them into manageable sub-problems, each handled by a specialized neural network module.

