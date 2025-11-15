# I-have-a-dreamer: Complete Architecture Documentation

## High-Level Overview

**I-have-a-dreamer** is a modular deep learning framework designed for sequential decision-making tasks on grid-based environments (specifically ARC - Abstraction and Reasoning Corpus). The system learns to predict intermediate steps in a multi-stage transformation process by decomposing the problem into several specialized prediction modules.

### Core Concept

The system models sequential transformations as a pipeline:
1. **Color Selection**: Given a state and a color selection action, predict which color will be selected
2. **Selection Mask**: Given a state, selection action, and predicted color, predict which regions will be selected
3. **Next State**: Given a state, transform action, and predicted selection mask, predict the next state
4. **Reward**: Given current state, next state, and target state, predict the reward
5. **Continuation**: Given current and next states, predict whether the episode continues

---

## System Architecture

### 1. Data Layer (`src/data/`)

#### ReplayBufferDataset
- **Purpose**: Loads and manages replay buffer data from `.pt`, `.pkl`, or `.hdf5` files
- **Key Features**:
  - Supports "fast tensor mode" for efficient loading (all fields pre-converted to tensors)
  - Three modes: `color_only`, `selection_color`, `end_to_end`
  - Handles variable-sized grids with padding (-1 values)
  - Extracts rich metadata: shape dimensions, color statistics, transition types

**Data Structure** (per transition):
```python
{
    'state': (H, W) grid,                    # Current grid state
    'next_state': (H, W) grid,              # Next grid state  
    'target_state': (H, W) grid,            # Final target state
    'action_colour': int,                    # Color selection action index
    'action_selection': int,                 # Selection action index
    'action_transform': int,                 # Transform action index
    'colour': int,                           # Ground truth selected color
    'selection_mask': (H, W) grid,          # Ground truth selection mask
    'reward': float,                         # Reward signal
    'done': bool,                            # Episode termination flag
    'shape_h', 'shape_w': int,               # Grid dimensions
    'num_colors_grid': int,                  # Number of unique colors
    'most_present_color': int,                # Most common color
    'least_present_color': int,              # Least common color
    # ... (similar fields for next_state and target_state)
}
```

---

### 2. State Encoding (`src/models/state_encoder.py`)

#### StateEncoder (Vision Transformer-based)
- **Architecture**: ViT-style transformer encoder
- **Input**: Grid state (H×W) with integer color values [-1, 0..vocab_size-2]
- **Output**: Latent vector (B, latent_dim)

**Processing Pipeline**:
1. **Token Embedding**: Each grid cell becomes a token via color embedding
2. **Positional Encoding**: Adds row/column positional embeddings
3. **Special Tokens**: Adds CLS token + 5 metadata tokens:
   - Row shape token (grid height)
   - Column shape token (grid width)
   - Most common color token
   - Least common color token
   - Unique color count token
4. **Transformer Blocks**: Pre-norm transformer layers with self-attention
5. **Pooling**: Uses CLS token as final representation

**Key Features**:
- Handles variable-sized grids via padding masks
- Incorporates grid statistics as learnable tokens
- Supports scaled or learned positional embeddings

---

### 3. Action Embedding (`src/models/action_embed.py`)

#### ActionEmbedder
- **Purpose**: Converts discrete action indices or one-hot vectors into dense embeddings
- **Three Types**:
  - `action_color_embedder`: Embeds color selection actions (23 actions → 32 dim)
  - `action_selection_embedder`: Embeds selection actions (6 actions → 32 dim)
  - `action_transform_embedder`: Embeds transform actions (18 actions → 32 dim)

**Why Embeddings?**
- Learnable semantic representations of actions
- More efficient than one-hot encoding
- Enables the model to learn action similarities

---

### 4. Prediction Modules

#### A. Color Predictor (`src/models/predictors/color_predictor.py`)

**Architecture**: Simple MLP
- **Input**: 
  - State latent vector (B, latent_dim)
  - Embedded color action (B, action_embed_dim)
- **Output**: Color logits (B, num_arc_colors=12)
- **Loss**: Cross-entropy

**Function**: Predicts which color will be selected given a state and color selection action.

---

#### B. Selection Mask Predictor (`src/models/predictors/selection_mask_predictor.py`)

**Architecture**: Transformer-based
- **Input**:
  - State latent vector (B, state_dim)
  - Embedded selection action (B, selection_embed_dim)
  - Color prediction distribution (B, num_arc_colors) - softmax of color logits
- **Output**: Predicted latent mask (B, latent_mask_dim)
- **Loss**: MSE or VICReg (configurable)

**Processing**:
1. Projects state and `[selection_action, color_pred]` to same dimension
2. Stacks as sequence of length 2
3. Applies transformer with positional encoding
4. Outputs latent representation of selection mask

**Ground Truth**: Selection mask is encoded via `MaskEncoder` (ViT-style) to produce target latent mask.

---

#### C. Next State Predictor (`src/models/predictors/next_state_predictor.py`)

**Architecture**: Transformer-based
- **Input**:
  - State latent vector (B, state_dim)
  - Embedded transform action (B, num_transform_actions) - one-hot
  - Predicted latent mask (B, latent_mask_dim)
- **Output**: Predicted next state latent (B, latent_dim)
- **Loss**: MSE or VICReg (configurable)

**Processing**:
1. Projects state and `[transform_action, latent_mask]` to same dimension
2. Stacks as sequence of length 2
3. Applies transformer with positional encoding
4. Uses first token (state) as output

**Function**: Predicts the next state embedding after applying a transformation.

---

#### D. Reward Predictor (`src/models/predictors/reward_predictor.py`)

**Architecture**: Simple MLP (3-layer)
- **Input**:
  - Current state latent (B, latent_dim)
  - Next state latent (B, latent_dim)
  - Target state latent (B, latent_dim)
- **Output**: Scalar reward prediction (B, 1)
- **Loss**: MSE

**Processing**:
1. Concatenates all three latent vectors
2. Passes through MLP layers
3. Outputs scalar reward

**Function**: Predicts reward signal based on progress toward target state.

---

#### E. Continuation Predictor (`src/models/predictors/continuation_predictor.py`)

**Architecture**: Similar to reward predictor (MLP)
- **Input**:
  - Current state latent (B, latent_dim)
  - Next state latent (B, latent_dim)
- **Output**: Continuation probability (B, 1) via sigmoid
- **Loss**: Binary cross-entropy

**Function**: Predicts whether the episode should continue (1-done).

---

### 5. Auxiliary Modules

#### Mask Encoder (`src/models/mask_encoder_new.py`)
- **Purpose**: Encodes ground truth selection masks into latent space
- **Architecture**: ViT-style transformer (similar to StateEncoder)
- **Output**: Latent mask representation (B, latent_mask_dim)
- **Used for**: Computing target latent mask for selection mask predictor training

#### State Decoder (`src/models/state_decoder.py`)
- **Purpose**: Decodes latent vectors back to grid states
- **Used for**: Autoencoder training and optional decoder-based losses
- **Output**: Grid logits + shape/statistics predictions

#### Mask Decoder (`src/models/mask_decoder_new.py`)
- **Purpose**: Decodes latent masks back to selection masks
- **Used for**: Optional decoder-based losses (alternative to latent space losses)

---

## Training Pipeline

### Training Scripts

#### 1. `train_autoencoder.py`
- **Purpose**: Pre-train state encoder-decoder pair
- **Loss**: Multi-component reconstruction loss:
  - Grid reconstruction (cross-entropy)
  - Shape prediction (height/width)
  - Color statistics prediction
- **Output**: Pre-trained encoder weights

#### 2. `train_color_predictor.py`
- **Purpose**: Train color prediction module
- **Mode**: `color_only` (doesn't need next_state)
- **Components**: StateEncoder + ActionEmbedder + ColorPredictor
- **Features**: Supports pretrained encoder loading, freezing

#### 3. `train_selection_predictor.py`
- **Purpose**: Train selection mask prediction (and color prediction jointly)
- **Mode**: `selection_color` (needs selection_mask but not next_state)
- **Components**: StateEncoder + ColorPredictor + MaskEncoder + SelectionMaskPredictor
- **Features**: 
  - Ground truth switch: use ground truth color vs predicted color
  - Decoder loss switch: use decoder loss vs latent space loss
  - VICReg loss option

#### 4. `train_next_state_predictor.py`
- **Purpose**: Train next state prediction (and all previous modules)
- **Mode**: `end_to_end` (needs next_state)
- **Components**: All modules except reward/continuation predictors
- **Features**: 
  - Can use ground truth vs predicted inputs
  - Decoder loss option
  - EMA target encoder for stability

#### 5. `train_reward_predictor.py`
- **Purpose**: Train reward prediction module
- **Mode**: `end_to_end`
- **Components**: StateEncoder + RewardPredictor
- **Features**: 
  - EMA target encoder
  - R² score tracking
  - Visualization plots

#### 6. `train_full_model.py`
- **Purpose**: End-to-end training of all modules simultaneously
- **Mode**: `end_to_end`
- **Components**: All modules
- **Loss**: Weighted sum of all individual losses

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT DATA                              │
│  State (H×W) + Actions + Target State + Metadata                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STATE ENCODER (ViT)                          │
│  Grid → Tokens → Positional Encoding → Transformer → Latent     │
│  Output: z_t (B, latent_dim)                                    │
└────────────────────────────┬────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
┌──────────────┐   ┌───────────────────┐   ┌──────────────┐
│ COLOR        │   │ SELECTION MASK    │   │ NEXT STATE   │
│ PREDICTOR    │   │ PREDICTOR         │   │ PREDICTOR    │
│              │   │                   │   │              │
│ z_t +        │   │ z_t +             │   │ z_t +        │
│ action_color │   │ action_selection +│   │ action_trans │
│ → color      │   │ color_pred        │   │ + mask_pred  │
│              │   │ → mask_latent     │   │ → z_{t+1}    │
└──────┬───────┘   └────────┬──────────┘   └──────┬───────┘
       │                    │                     │
       │                    │                     │
       └────────────────────┼─────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │   REWARD PREDICTOR    │
                │                       │
                │ z_t + z_{t+1} +       │
                │ z_target → reward     │
                └───────────────────────┘
                            │
                            ▼
                ┌───────────────────────┐
                │ CONTINUATION PREDICTOR│
                │                       │
                │ z_t + z_{t+1} →       │
                │ continuation_prob     │
                └───────────────────────┘
```

---

## Detailed Module Interaction Flow

### Forward Pass (Training)

```
1. INPUT PREPARATION
   ├─ Load batch: state, actions, targets, metadata
   └─ Move to device (MPS/CUDA/CPU)

2. STATE ENCODING
   ├─ Encode current state → z_t
   ├─ Encode next state → z_{t+1} (for training)
   └─ Encode target state → z_target (for reward)

3. ACTION EMBEDDING
   ├─ Embed color action → action_color_emb
   ├─ Embed selection action → action_selection_emb
   └─ One-hot transform action → action_transform_onehot

4. COLOR PREDICTION
   ├─ Concatenate [z_t, action_color_emb]
   ├─ ColorPredictor → color_logits
   └─ Softmax → color_distribution

5. SELECTION MASK PREDICTION
   ├─ Option A: Use ground truth color (if use_ground_truth=True)
   │  └─ One-hot encode target_color
   └─ Option B: Use predicted color_distribution
   ├─ Concatenate [action_selection_emb, color_input]
   ├─ SelectionMaskPredictor → pred_mask_latent
   ├─ Encode ground truth mask → target_mask_latent
   └─ Compute loss (MSE or VICReg)

6. NEXT STATE PREDICTION
   ├─ Concatenate [action_transform_onehot, pred_mask_latent]
   ├─ NextStatePredictor → pred_z_{t+1}
   ├─ Compare with ground truth z_{t+1}
   └─ Compute loss (MSE or VICReg)

7. REWARD PREDICTION
   ├─ RewardPredictor(z_t, pred_z_{t+1}, z_target)
   ├─ Output: pred_reward
   └─ Compute MSE loss vs ground truth reward

8. CONTINUATION PREDICTION
   ├─ ContinuationPredictor(z_t, pred_z_{t+1})
   ├─ Output: continuation_prob
   └─ Compute BCE loss vs (1 - done)

9. BACKWARD PASS
   ├─ Sum all losses (weighted)
   ├─ Backward propagation
   ├─ Gradient clipping
   └─ Optimizer step
```

---

## Configuration System

### `config.yaml` Structure

```yaml
# Data
buffer_path: "data/buffer_all_solved_100.pt"

# Training Switches
use_ground_truth: false          # Use GT inputs vs predicted
use_decoder_loss: true           # Use decoder losses vs latent losses

# State Encoder
encoder_type: "vit"
latent_dim: 256
use_pretrained_encoder: true
pretrained_encoder_path: "weights/best_model_next_state_predictor.pth"
freeze_pretrained_encoder: false

encoder_params:
  depth: 4
  heads: 16
  mlp_dim: 512
  transformer_dim: 96
  image_size: [10, 10]
  input_channels: 1
  colors_vocab_size: 12

# Action Embedders
action_embedders:
  action_color_embedder:
    num_actions: 23
    embed_dim: 32
  action_selection_embedder:
    num_actions: 6
    embed_dim: 32
  action_transform_embedder:
    num_actions: 18
    embed_dim: 32

# Module Configurations
color_predictor:
  hidden_dim: 256

selection_mask:
  latent_mask_dim: 256
  mask_encoder_params: {...}
  mask_predictor_params: {...}
  use_vicreg: false

next_state:
  latent_mask_dim: 0
  transformer_depth: 2
  transformer_heads: 2
  use_vicreg: false

reward_predictor:
  latent_dim: 256
  hidden_dim: 256
  num_layers: 3

# Training Parameters
batch_size: 96
num_epochs: 200
learning_rate: 0.0003
gradient_clip_norm: 1.0
early_stopping_patience: 10
```

---

## Key Design Patterns

### 1. Modular Architecture
- Each predictor is independent and can be trained separately
- Shared state encoder across all modules
- Easy to add/remove modules

### 2. Progressive Training
- Start with autoencoder (unsupervised)
- Train color predictor (simplest task)
- Train selection predictor (depends on color)
- Train next state predictor (depends on selection)
- Train reward/continuation (depend on next state)

### 3. Flexible Loss Functions
- Latent space losses (MSE/VICReg) for faster training
- Decoder losses for better reconstruction quality
- Configurable via switches

### 4. Ground Truth vs Predicted Inputs
- Can use ground truth intermediate predictions during training
- Gradually switch to predicted inputs for better generalization
- Helps with training stability

### 5. EMA Target Encoder
- Used in reward/next state training
- Provides stable targets for learning
- Reduces training instability

---

## Loss Functions

### 1. Color Prediction Loss
- **Type**: Cross-Entropy
- **Target**: Ground truth color class

### 2. Selection Mask Loss
- **Option A**: MSE between predicted and target latent masks
- **Option B**: VICReg loss (similarity + variance + covariance)
- **Option C**: Decoder loss (cross-entropy on reconstructed mask)

### 3. Next State Loss
- **Option A**: MSE between predicted and target latent states
- **Option B**: VICReg loss
- **Option C**: Decoder loss (reconstruction loss)

### 4. Reward Loss
- **Type**: MSE
- **Target**: Ground truth reward scalar

### 5. Continuation Loss
- **Type**: Binary Cross-Entropy
- **Target**: (1 - done) flag

---

## Training Modes

### Mode 1: `color_only`
- **Used by**: `train_color_predictor.py`
- **Requires**: state, action_colour, colour
- **Doesn't need**: next_state, selection_mask

### Mode 2: `selection_color`
- **Used by**: `train_selection_predictor.py`
- **Requires**: state, actions, selection_mask, colour
- **Doesn't need**: next_state

### Mode 3: `end_to_end`
- **Used by**: `train_next_state_predictor.py`, `train_reward_predictor.py`, `train_full_model.py`
- **Requires**: All fields including next_state

---

## File Organization

```
I-have-a-dreamer/
├── config.yaml                    # Main configuration
├── config_autoencoder.yaml         # Autoencoder config
├── train_*.py                      # Training scripts
├── src/
│   ├── data/
│   │   ├── replay_buffer_dataset.py    # Dataset loader
│   │   └── inspect_replay_buffer.py     # Buffer inspection
│   ├── models/
│   │   ├── state_encoder.py            # State encoder (ViT)
│   │   ├── state_decoder.py            # State decoder
│   │   ├── mask_encoder_new.py         # Mask encoder
│   │   ├── mask_decoder_new.py         # Mask decoder
│   │   ├── action_embed.py             # Action embedders
│   │   └── predictors/
│   │       ├── color_predictor.py
│   │       ├── selection_mask_predictor.py
│   │       ├── next_state_predictor.py
│   │       ├── reward_predictor.py
│   │       └── continuation_predictor.py
│   ├── losses/
│   │   ├── vicreg.py                   # VICReg loss
│   │   ├── barlow_twins.py
│   │   └── dino.py
│   └── utils/
│       └── weight_init.py              # Weight initialization
├── data/                              # Buffer files (.pt)
├── weights/                            # Saved model checkpoints
└── docs/                               # Documentation
```

---

## Key Features & Innovations

1. **Hierarchical Prediction**: Breaks complex transformation into simpler sub-tasks
2. **Transformer-Based Architecture**: Uses attention mechanisms for better feature learning
3. **Rich State Representation**: Incorporates grid statistics as learnable tokens
4. **Flexible Training**: Supports multiple training strategies (progressive, end-to-end)
5. **Robust Loss Functions**: VICReg for better latent space learning
6. **Action Embeddings**: Learnable action representations instead of one-hot
7. **Decoder-Based Supervision**: Optional reconstruction losses for better quality

---

## Usage Workflow

1. **Prepare Data**: Create replay buffer with transitions
2. **Pre-train Encoder**: Run `train_autoencoder.py` (optional)
3. **Train Modules Sequentially**:
   - `train_color_predictor.py`
   - `train_selection_predictor.py`
   - `train_next_state_predictor.py`
   - `train_reward_predictor.py`
4. **Or Train End-to-End**: `train_full_model.py`
5. **Evaluate**: Use saved checkpoints for inference

---

## Summary

This repository implements a sophisticated modular system for learning sequential decision-making on grid-based environments. By decomposing the problem into specialized prediction modules and using modern transformer architectures, it achieves effective learning of complex multi-step transformations. The design emphasizes flexibility, modularity, and progressive training strategies.

