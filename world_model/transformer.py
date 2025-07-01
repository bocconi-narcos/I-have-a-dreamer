class DecoderTransformerTorch(nn.Module):
    '''
    PyTorch re-implementation of the Flax-based DecoderTransformer.
    Now only uses state embedding as input.
    '''
    def __init__(self, config: DecoderTransformerConfig, emb_state_dim: int):
        super().__init__()
        self.config = config
        
        # Project state embedding to full emb_dim
        self.emb_state_proj = nn.Linear(emb_state_dim, config.emb_dim).to(DEVICE)
    
        # Create transformer layers using the unified config.
        self.layers = nn.ModuleList([TransformerLayer(config).to(DEVICE) for _ in range(config.num_layers)])
        self.layer_norm = nn.LayerNorm(config.emb_dim).to(DEVICE)
        
        # Projections to logits.
        self.shape_row_proj = nn.Linear(config.emb_dim, config.max_rows).to(DEVICE)
        self.shape_col_proj = nn.Linear(config.emb_dim, config.max_cols).to(DEVICE)
        # grid_proj now outputs max_rows * max_cols * vocab_size logits.
        self.grid_proj = nn.Linear(config.emb_dim, config.max_rows * config.max_cols * config.vocab_size).to(DEVICE)

    def forward(self, embedded_state, dropout_eval: bool):
        '''
        Args:
            embedded_state: shape (B, emb_state_dim)
            dropout_eval: bool, if True, disables dropout for evaluation mode
            
        Returns:
            shape_row_logits: shape (B, R), the logits for grid shape row
            shape_col_logits: shape (B, C), the logits for grid shape col
            grid_logits: shape (B, R*C, vocab_size), the logits for grid tokens
        '''
        assert len(embedded_state.shape) == 2

        # Project state embedding to full emb_dim.
        x = self.emb_state_proj(embedded_state)
        print("DEBUG: after state projection, x shape:", x.shape)

        # Pass through transformer layers.
        for i, layer in enumerate(self.layers):
            x = layer(x, dropout_eval=dropout_eval)
            print(f"DEBUG: after transformer layer {i}, x shape:", x.shape)
        
        # Apply layer normalization.
        x = self.layer_norm(x)
        print("DEBUG: after layer normalization, x shape:", x.shape)
        
        # Project to logits.
        shape_row_logits = self.shape_row_proj(x)
        shape_col_logits = self.shape_col_proj(x)
        grid_logits = self.grid_proj(x)  # (B, max_rows * max_cols * vocab_size)
        print("DEBUG: after grid projection, grid_logits shape:", grid_logits.shape)
        
        B = grid_logits.size(0)
        grid_logits = grid_logits.view(B, self.config.max_rows * self.config.max_cols, self.config.vocab_size)
        print("DEBUG: after reshaping grid_logits, grid_logits shape:", grid_logits.shape)
        
        print("DEBUG: shape_row_logits shape:", shape_row_logits.shape)
        print("DEBUG: shape_col_logits shape:", shape_col_logits.shape)
        
        return shape_row_logits, shape_col_logits, grid_logits 