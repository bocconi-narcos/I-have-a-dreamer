# Color Predictor - Nuova Architettura Design

## Requisiti

### Inputs:
1. **Action Embedding**: `[B, latent_dim]` - embedding dell'azione
2. **State Tokens**: `[B, num_token, latent_dim]` - sequenza di token dallo state encoder
3. **Causal Mask**: maschera che indica quali token sono utili e quali sono padding

### Output:
- **Color Logits**: `[B, num_colors]` - logits per classificazione colore

### Architettura:
- Usare "some attention layers"
- Probabilmente cross-attention tra action e tokens
- Classificare con final prediction head
- Production-ready, well-reasoned
- Non usare attention con cose inutili (usare causal mask)
- Profondità necessaria per astrazione
- Altamente customizzabile (es. num_layer)

---

## Design Proposto

### Architettura: Cross-Attention Color Predictor

```
Input:
  - action_embedding: (B, latent_dim)
  - state_tokens: (B, num_token, latent_dim)
  - causal_mask: (B, num_token, num_token) o (B, num_token)

Architecture:
  1. Action Projection (se necessario)
     action_embedding -> (B, latent_dim)
  
  2. Cross-Attention Layers (N layers)
     For each layer:
       - Query: action_embedding (B, 1, latent_dim) [expand]
       - Key: state_tokens (B, num_token, latent_dim)
       - Value: state_tokens (B, num_token, latent_dim)
       - Mask: causal_mask -> padding_mask (B, num_token)
       - Output: (B, 1, latent_dim)
  
  3. Final MLP Head
     pooled_action -> (B, latent_dim) -> (B, num_colors)

Output:
  - color_logits: (B, num_colors)
```

### Componenti:

1. **PreNormCrossAttentionBlock**
   - Pre-norm architecture (più stabile)
   - Cross-attention: action query, state tokens key/value
   - Feed-forward network
   - Residual connections

2. **Mask Handling**
   - Convert causal_mask (B, seq_len, seq_len) -> padding_mask (B, seq_len)
   - padding_mask[i] = True se token i è padding/invalido
   - Usare key_padding_mask in MultiheadAttention

3. **Final Prediction Head**
   - Pooling (se necessario) o uso diretto
   - MLP: latent_dim -> hidden_dim -> num_colors

---

## Scelte Architetturali

### 1. Pre-norm vs Post-norm
✅ **Pre-norm**: Più stabile, usato nel resto del codicebase

### 2. Cross-Attention vs Self-Attention
✅ **Cross-Attention**: Action come query, state tokens come key/value
- Permette all'action di "interrogare" i token dello state
- Più efficiente di concatenare tutto

### 3. Multi-layer vs Single-layer
✅ **Multi-layer**: Permette astrazione profonda
- Ogni layer può apprendere diversi livelli di astrazione
- Configurabile con `num_layers`

### 4. Mask Handling
✅ **Padding Mask**: Convert causal_mask in padding_mask
- MultiheadAttention usa `key_padding_mask`
- True = mask out (padding)

### 5. Final Pooling
✅ **Direct Use**: Usare direttamente l'action dopo cross-attention
- L'action è già un singolo token (B, 1, latent_dim)
- Non serve pooling aggiuntivo

---

## Implementazione

### Classe: CrossAttentionColorPredictor

```python
class PreNormCrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention block"""
    def __init__(self, latent_dim, heads, mlp_dim, dropout):
        # Cross-attention: action query, state tokens key/value
        # Feed-forward
        # Residual connections

class CrossAttentionColorPredictor(nn.Module):
    """Cross-attention based color predictor"""
    def __init__(
        self,
        latent_dim: int,
        num_colors: int = 11,
        num_layers: int = 2,  # Customizzabile
        heads: int = 8,
        mlp_dim: int = 256,
        dropout: float = 0.1,
        mlp_hidden_dim: int = 128
    ):
        # N cross-attention layers
        # Final MLP head
    
    def forward(
        self,
        action_embedding: torch.Tensor,  # (B, latent_dim)
        state_tokens: torch.Tensor,      # (B, num_token, latent_dim)
        causal_mask: torch.Tensor = None # (B, num_token, num_token) or (B, num_token)
    ) -> torch.Tensor:
        # Convert mask
        # Cross-attention layers
        # Final prediction
```

---

## Vantaggi

1. ✅ **Efficiente**: Cross-attention è più efficiente di concatenare tutto
2. ✅ **Stabile**: Pre-norm architecture
3. ✅ **Flessibile**: Usa causal mask per evitare attention su padding
4. ✅ **Profondo**: Multi-layer per astrazione
5. ✅ **Customizzabile**: Parametri configurabili
6. ✅ **Production-ready**: Architettura ben ragionata

---

## Compatibilità Backward

Per mantenere compatibilità con codice esistente:
- Mantenere `ColorPredictor` originale
- Aggiungere nuova classe `CrossAttentionColorPredictor`
- Aggiornare training script per usare nuova classe quando disponibili tokens

