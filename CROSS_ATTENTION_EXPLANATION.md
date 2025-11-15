# Cross-Attention vs Self-Attention - Spiegazione

## Panoramica

Il `CrossAttentionColorPredictor` usa **SOLO Cross-Attention**, NON Self-Attention.

---

## Cross-Attention vs Self-Attention

### Self-Attention
**Definizione**: Ogni elemento della sequenza attende a tutti gli altri elementi della **stessa sequenza**.

**Query, Key, Value**: Tutti dalla stessa sequenza
```
Input: x = [token1, token2, token3, ..., tokenN]
Query = x, Key = x, Value = x
```

**Esempio**:
```python
# Self-attention in StateEncoder
state_tokens = [metadata1, metadata2, ..., grid1, grid2, ...]
# Ogni token attende a tutti gli altri token dello state
attention(state_tokens, state_tokens, state_tokens)
```

**Quando si usa**:
- Quando vuoi che elementi della stessa sequenza interagiscano tra loro
- Esempio: `StateEncoder` usa self-attention per far interagire i token dello state

### Cross-Attention
**Definizione**: Una sequenza (query) attende a un'altra sequenza (key/value).

**Query, Key, Value**: Da sequenze diverse
```
Query: sequenza A = [q1, q2, ...]
Key/Value: sequenza B = [k1, k2, ..., v1, v2, ...]
```

**Esempio**:
```python
# Cross-attention in ColorPredictor
action_embedding = [action]  # Query
state_tokens = [metadata1, metadata2, ..., grid1, ...]  # Key/Value
# Action attende ai token dello state
attention(action_embedding, state_tokens, state_tokens)
```

**Quando si usa**:
- Quando vuoi che una sequenza "interroghi" un'altra sequenza
- Esempio: `ColorPredictor` usa cross-attention per far sì che l'action "interroghi" i token dello state

---

## Nel Nostro Color Predictor

### ✅ Cross-Attention (USATO)

**Dove**: `PreNormCrossAttentionBlock` (linea 176-179)

```python
# Cross-attention: action (query) attende a state tokens (key/value)
attn_out, _ = self.cross_attn(
    query_norm,      # Query: action embedding (B, 1, latent_dim)
    kv_norm,         # Key: state tokens (B, num_token, latent_dim)
    kv_norm,         # Value: state tokens (B, num_token, latent_dim)
    key_padding_mask=key_padding_mask
)
```

**Spiegazione**:
- **Query**: Action embedding `(B, 1, latent_dim)` - "Cosa voglio sapere?"
- **Key**: State tokens `(B, num_token, latent_dim)` - "Dove cercare?"
- **Value**: State tokens `(B, num_token, latent_dim)` - "Cosa prendere?"

**Risultato**: L'action embedding viene aggiornato con informazioni dai token dello state.

### ❌ Self-Attention (NON USATO)

**Perché non usiamo self-attention**:
- L'action è un singolo token, non ha senso farlo attendere a se stesso
- I token dello state già interagiscono tra loro nel `StateEncoder`
- Vogliamo che l'action "interroghi" lo state, non che interagisca con se stesso

---

## Confronto Visivo

### Self-Attention (StateEncoder)
```
State Tokens: [metadata1, metadata2, grid1, grid2, ...]
                ↓         ↓         ↓       ↓
              ┌─────────────────────────────┐
              │  Self-Attention              │
              │  Ogni token attende a tutti  │
              └─────────────────────────────┘
                ↓         ↓         ↓       ↓
         Updated Tokens: [m1', m2', g1', g2', ...]
```

### Cross-Attention (ColorPredictor)
```
Action: [action_emb]
         ↓
    ┌─────────────────────────────┐
    │  Cross-Attention             │
    │  Action (query) attende a   │
    │  State Tokens (key/value)    │
    └─────────────────────────────┘
         ↓
State: [metadata1, metadata2, grid1, grid2, ...]
         ↓
    Updated Action: [action_emb']
```

---

## Perché Cross-Attention nel Color Predictor?

### Vantaggi:

1. **Efficienza**
   - Action è un singolo token, non serve self-attention
   - Cross-attention permette all'action di "interrogare" solo i token rilevanti

2. **Semantica Corretta**
   - Action vuole sapere informazioni dallo state
   - Cross-attention modella questa relazione query→source

3. **Flessibilità**
   - Action può apprendere quali token dello state sono rilevanti
   - Causal mask evita attention su token padding

4. **Astrazione Profonda**
   - Multi-layer cross-attention permette astrazione a diversi livelli
   - Ogni layer può apprendere pattern diversi

---

## Esempio Concreto

### Input:
```python
action_embedding = [0.2, 0.5, ..., 0.1]  # (B, latent_dim) - "Seleziona colore rosso"
state_tokens = [
    [row_shape],      # metadata token 1
    [col_shape],      # metadata token 2
    [most_color],     # metadata token 3
    [grid_pos_1],    # grid token 1
    [grid_pos_2],    # grid token 2
    ...
]  # (B, num_token, latent_dim)
```

### Cross-Attention Process:
```
1. Action (query) calcola similarity con ogni token dello state (key)
2. Ottiene attention weights: [w1, w2, w3, w4, w5, ...]
3. Pesata dei token dello state (value) con attention weights
4. Aggiorna action con informazioni rilevanti dallo state
```

### Risultato:
```python
updated_action = action + weighted_sum(state_tokens)
# Action ora contiene informazioni rilevanti dallo state
# per predire il colore corretto
```

---

## Confronto con Altri Modelli

### StateEncoder (Self-Attention)
```python
# Self-attention: token dello state interagiscono tra loro
out = self_attention(state_tokens, state_tokens, state_tokens)
```

### ColorPredictor (Cross-Attention)
```python
# Cross-attention: action interroga token dello state
out = cross_attention(action, state_tokens, state_tokens)
```

### Perché Diversi?
- **StateEncoder**: Deve far interagire i token dello state tra loro per creare rappresentazione coerente
- **ColorPredictor**: Deve permettere all'action di estrarre informazioni rilevanti dallo state

---

## Riepilogo

| Aspetto | Self-Attention | Cross-Attention |
|---------|---------------|-----------------|
| **Query** | Stessa sequenza | Sequenza diversa |
| **Key/Value** | Stessa sequenza | Sequenza diversa |
| **Uso** | Interazione interna | Interrogazione esterna |
| **Esempio** | StateEncoder | ColorPredictor |
| **Quando** | Elementi devono interagire | Una sequenza interroga l'altra |

### Nel Nostro Caso:
- ✅ **Cross-Attention**: Action (query) → State Tokens (key/value)
- ❌ **Self-Attention**: NON usato (action è singolo token)

---

## Codice di Riferimento

### Cross-Attention Implementation
```python
# Linea 176-179 in PreNormCrossAttentionBlock
attn_out, _ = self.cross_attn(
    query_norm,      # Action embedding (B, 1, latent_dim)
    kv_norm,         # State tokens (B, num_token, latent_dim)
    kv_norm,         # State tokens (B, num_token, latent_dim)
    key_padding_mask=key_padding_mask
)
```

### Self-Attention (per confronto, in StateEncoder)
```python
# StateEncoder usa self-attention
attn_out, _ = self.attn(
    x_norm,          # State tokens (B, seq_len, emb_dim)
    x_norm,          # State tokens (B, seq_len, emb_dim)
    x_norm,          # State tokens (B, seq_len, emb_dim)
    key_padding_mask=src_key_padding_mask
)
```

---

**Conclusione**: Il `CrossAttentionColorPredictor` usa **SOLO Cross-Attention**, permettendo all'action di "interrogare" i token dello state per predire il colore corretto.

