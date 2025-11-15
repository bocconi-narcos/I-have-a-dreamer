# Spiegazione Dettagliata del Wrapper per Compatibilità

## Perché è Necessario il Wrapper?

### Problema: Breaking Change

**Prima delle modifiche**:
```python
encoder = StateEncoder(...)
latent = encoder(x, shape_h, shape_w, ...)  
# latent: (B, latent_dim) - singolo vettore
```

**Dopo le modifiche**:
```python
encoder = StateEncoder(...)
tokens, causal_mask = encoder(x, shape_h, shape_w, ...)
# tokens: (B, seq_len, latent_dim) - tutti i token
# causal_mask: (B, seq_len, seq_len) - maschera attention
```

**Conseguenza**: Tutto il codice esistente che usa `StateEncoder` si romperebbe perché si aspetta un singolo vettore, non una tupla.

### Soluzione: Wrapper Opzionale

Il wrapper permette al codice esistente di continuare a funzionare **senza modifiche**, mentre il nuovo codice può usare direttamente `StateEncoder` per accedere a tutti i token.

---

## Come Funziona il Wrapper

### Architettura

```
StateEncoderWrapper
    │
    ├── Avvolge: StateEncoder (encoder interno)
    │
    ├── Forward pass:
    │   1. Chiama encoder.forward() → ottiene (tokens, causal_mask)
    │   2. Applica pooling ai token → ottiene (B, latent_dim)
    │   3. Restituisce solo il vettore pooled
    │
    └── Attributi: Inoltra accesso agli attributi dell'encoder interno
```

### Implementazione Dettagliata

#### 1. Inizializzazione

```python
class StateEncoderWrapper(nn.Module):
    def __init__(self, encoder: StateEncoder, pool_method: str = 'mean'):
        super().__init__()
        # Registra encoder come submodule PyTorch
        self.add_module('encoder', encoder)
        self.pool_method = pool_method  # 'mean' o 'first'
        
        # Copia attributi importanti per compatibilità
        self.latent_dim = encoder.latent_dim
        self.emb_dim = encoder.emb_dim
        self.max_rows = encoder.max_rows
        self.max_cols = encoder.max_cols
```

**Spiegazione**:
- `add_module()`: Registra l'encoder come submodule PyTorch (necessario per evitare problemi con `__getattr__`)
- `pool_method`: Sceglie come aggregare i token (`'mean'` = media, `'first'` = primo token)
- Attributi copiati: Permettono accesso diretto senza dover passare attraverso `__getattr__`

#### 2. Forward Pass

```python
def forward(self, x, shape_h=None, shape_w=None, ...):
    # Gestisce argomenti opzionali per massima compatibilità
    if shape_h is None:
        # Crea valori di default basati su dimensioni input
        ...
    
    # Chiama encoder interno
    encoder = self._modules['encoder']  # Accesso diretto per evitare ricorsione
    tokens, causal_mask = encoder(x, shape_h, shape_w, ...)
    
    # Pooling: trasforma (B, seq_len, latent_dim) → (B, latent_dim)
    return encoder.pool_tokens(tokens, causal_mask, method=self.pool_method)
```

**Flusso Dati**:
```
Input: x, shape_h, shape_w, ...
    ↓
StateEncoder.forward()
    ↓
Output: tokens (B, seq_len, latent_dim), causal_mask (B, seq_len, seq_len)
    ↓
pool_tokens() con metodo scelto
    ↓
Output: latent (B, latent_dim) ← Compatibile con codice esistente!
```

#### 3. Accesso Attributi

```python
def __getattr__(self, name):
    """Inoltra accesso attributi all'encoder interno"""
    try:
        return super().__getattribute__(name)
    except AttributeError:
        # Accesso via _modules per evitare ricorsione infinita
        encoder = self._modules.get('encoder')
        if encoder is not None:
            return getattr(encoder, name)
        raise AttributeError(...)
```

**Perché necessario**: 
- PyTorch usa `__getattr__` internamente per accedere a parametri/moduli
- Senza questa gestione, si creerebbe ricorsione infinita quando si accede a `self.encoder`
- Usando `_modules` si accede direttamente senza triggerare `__getattr__`

---

## Metodi di Pooling Disponibili

### 1. Mean Pooling (`method='mean'`)

**Cosa fa**: Calcola la media di tutti i token lungo la dimensione sequenza

```python
# tokens: (B, seq_len, latent_dim)
# Esempio: B=2, seq_len=105, latent_dim=256

pooled = tokens.mean(dim=1)
# pooled: (2, 256)
# Ogni elemento è la media di 105 token
```

**Vantaggi**:
- Usa tutte le informazioni disponibili
- Più robusto (meno sensibile a singoli token)

**Quando usare**: Quando vuoi incorporare informazioni da tutti i token

### 2. First Token Pooling (`method='first'`)

**Cosa fa**: Usa solo il primo token (primo metadata token)

```python
# tokens: (B, seq_len, latent_dim)
pooled = tokens[:, 0, :]
# pooled: (B, latent_dim)
# Usa solo il primo token (shape_h token)
```

**Vantaggi**:
- Più veloce (nessun calcolo)
- Comportamento simile al vecchio CLS token

**Quando usare**: Quando vuoi comportamento simile al CLS originale

---

## Esempi di Utilizzo

### Esempio 1: Codice Esistente (Nessuna Modifica)

**Prima** (funziona ancora):
```python
from src.models.state_encoder import StateEncoder, StateEncoderWrapper

# Vecchio codice - continua a funzionare!
encoder = StateEncoderWrapper(
    StateEncoder(image_size=(10, 10), ...),
    pool_method='mean'
)

latent = encoder(x, shape_h, shape_w, ...)
# latent: (B, latent_dim) ← Stesso formato di prima!
```

**Dopo** (nuovo codice, migliore):
```python
from src.models.state_encoder import StateEncoder

# Nuovo codice - accesso a tutti i token
encoder = StateEncoder(image_size=(10, 10), ...)

tokens, causal_mask = encoder(x, shape_h, shape_w, ...)
# tokens: (B, seq_len, latent_dim) ← Più informazioni!
# causal_mask: (B, seq_len, seq_len) ← Utile per attention custom
```

### Esempio 2: Confronto Pooling Methods

```python
encoder_base = StateEncoder(...)

# Mean pooling (default)
wrapper_mean = StateEncoderWrapper(encoder_base, pool_method='mean')
latent_mean = wrapper_mean(...)  # Media di tutti i token

# First token pooling
wrapper_first = StateEncoderWrapper(encoder_base, pool_method='first')
latent_first = wrapper_first(...)  # Solo primo token

# I risultati saranno diversi!
assert not torch.allclose(latent_mean, latent_first)
```

### Esempio 3: Accesso Attributi

```python
wrapper = StateEncoderWrapper(StateEncoder(...))

# Accesso diretto agli attributi dell'encoder
print(wrapper.latent_dim)  # Funziona (copiato)
print(wrapper.max_rows)    # Funziona (copiato)
print(wrapper.emb_dim)    # Funziona (copiato)

# Accesso ad altri attributi (inoltro automatico)
print(wrapper.color_embed)  # Funziona (inoltro via __getattr__)
```

---

## Gestione Argomenti Opzionali

Il wrapper gestisce anche il caso in cui alcuni argomenti non vengano forniti:

```python
# Con tutti gli argomenti (come prima)
latent = wrapper(x, shape_h, shape_w, most_common, least_common, num_colors)

# Con argomenti opzionali (nuova funzionalità)
latent = wrapper(x)  # Crea valori di default automaticamente
```

**Come funziona**:
- Se `shape_h` è `None`, deduce dimensioni da `x`
- Crea valori di default per tutti gli altri parametri
- Permette massima flessibilità

---

## Vantaggi del Wrapper

### 1. **Zero Breaking Changes**
- Codice esistente continua a funzionare senza modifiche
- Migrazione graduale possibile

### 2. **Flessibilità**
- Scegli metodo di pooling (`mean` vs `first`)
- Accesso trasparente agli attributi dell'encoder

### 3. **Minimal Overhead**
- Pooling è operazione veloce (media o slicing)
- Nessuna copia di dati, solo view/operazioni in-place

### 4. **Production Ready**
- Gestisce edge cases (argomenti opzionali)
- Evita problemi di ricorsione con PyTorch
- Ben testato (3 test dedicati)

---

## Svantaggi / Limitazioni

### 1. **Perdita di Informazione**
- Pooling perde dettagli spaziali dei token individuali
- Non puoi accedere a token specifici della griglia

### 2. **Overhead Computazionale**
- Pooling aggiunge piccolo overhead (minimo)
- Due chiamate invece di una (encoder + pooling)

### 3. **Non Ottimale per Nuovo Codice**
- Se stai scrivendo nuovo codice, meglio usare direttamente `StateEncoder`
- Il wrapper è solo per backward compatibility

---

## Quando Usare il Wrapper

### ✅ Usa il Wrapper quando:
- Stai aggiornando codice esistente gradualmente
- Hai bisogno di compatibilità immediata senza modifiche
- Vuoi comportamento simile al vecchio CLS token

### ❌ NON usare il Wrapper quando:
- Stai scrivendo nuovo codice da zero
- Hai bisogno di accesso a token specifici
- Vuoi massime prestazioni (usa direttamente `StateEncoder`)

---

## Confronto: Con vs Senza Wrapper

### Con Wrapper (Backward Compatible)
```python
encoder = StateEncoderWrapper(StateEncoder(...))
latent = encoder(...)  # (B, latent_dim)

# Vantaggi:
# ✅ Codice esistente funziona
# ✅ Nessuna modifica necessaria
# ✅ Pooling automatico

# Svantaggi:
# ❌ Perdi informazioni sui token individuali
# ❌ Piccolo overhead computazionale
```

### Senza Wrapper (Nuovo Approccio)
```python
encoder = StateEncoder(...)
tokens, causal_mask = encoder(...)  
# tokens: (B, seq_len, latent_dim)
# causal_mask: (B, seq_len, seq_len)

# Vantaggi:
# ✅ Accesso a tutti i token
# ✅ Puoi usare causal_mask per attention custom
# ✅ Nessun overhead di pooling
# ✅ Più informazioni disponibili

# Svantaggi:
# ❌ Devi modificare codice esistente
# ❌ Devi gestire tuple invece di singolo vettore
```

---

## Esempio Pratico: Migrazione Graduale

### Fase 1: Usa Wrapper (Nessuna Modifica)
```python
# train_color_predictor.py - continua a funzionare
encoder = StateEncoderWrapper(StateEncoder(...))
latent = encoder(state, ...)  # (B, latent_dim)
color_logits = color_predictor(latent, action_emb)
```

### Fase 2: Migrazione Graduale (Opzionale)
```python
# Usa helper method per pooling manuale
encoder = StateEncoder(...)
tokens, causal_mask = encoder(state, ...)
latent = encoder.pool_tokens(tokens, causal_mask, method='mean')
color_logits = color_predictor(latent, action_emb)
```

### Fase 3: Nuovo Approccio (Massime Prestazioni)
```python
# Usa direttamente tutti i token
encoder = StateEncoder(...)
tokens, causal_mask = encoder(state, ...)
# tokens: (B, seq_len, latent_dim) - usa direttamente!
# Puoi fare attention sui token, selezionare token specifici, ecc.
```

---

## Test del Wrapper

Il wrapper è testato con 3 test dedicati:

1. **`test_wrapper_returns_single_tensor`**
   - Verifica che restituisca `(B, latent_dim)` invece di tuple
   - Confronta con comportamento originale

2. **`test_wrapper_attribute_access`**
   - Verifica che gli attributi siano accessibili
   - Testa inoltro attributi all'encoder interno

3. **`test_wrapper_different_pool_methods`**
   - Verifica che entrambi i metodi (`mean`, `first`) funzionino
   - Verifica che producano risultati diversi

---

## Riepilogo Tecnico

### Struttura Interna

```
StateEncoderWrapper
├── encoder: StateEncoder (submodule PyTorch)
├── pool_method: str ('mean' | 'first')
├── latent_dim: int (copiato)
├── emb_dim: int (copiato)
├── max_rows: int (copiato)
└── max_cols: int (copiato)
```

### Flusso Esecuzione

```
forward(x, ...)
    ↓
[Gestione argomenti opzionali]
    ↓
encoder.forward(x, ...) → (tokens, causal_mask)
    ↓
pool_tokens(tokens, causal_mask, method) → latent
    ↓
return latent (B, latent_dim)
```

### Gestione Attributi

```
accesso a attributo
    ↓
__getattr__(name)
    ↓
[Prova super().__getattribute__]
    ↓ (se fallisce)
[Accesso via _modules['encoder']]
    ↓
getattr(encoder, name)
```

---

## Conclusione

Il `StateEncoderWrapper` è una soluzione elegante per mantenere **compatibilità all'indietro** senza dover modificare tutto il codice esistente. Permette:

1. ✅ **Zero breaking changes** - Codice esistente funziona
2. ✅ **Migrazione graduale** - Puoi aggiornare quando vuoi
3. ✅ **Flessibilità** - Scegli metodo di pooling
4. ✅ **Production ready** - Ben testato e robusto

È un **ponte temporaneo** che permette al nuovo codice di usare tutti i token mentre il vecchio codice continua a funzionare con il formato originale.

