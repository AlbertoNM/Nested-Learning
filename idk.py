

class SimpleTransformer(nn.Module): 
    
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=2): 
        
        self.embedding = nn.Embedding(vocab_size, d_model) 
        self.pos_encoding = PositionalEncoding(d_model) 
        self.transformer_blocks = nn.ModuleList([ TransformerBlock(d_model, n_heads) for _ in range(n_layers) ]) 
        self.lm_head = nn.Linear(d_model, vocab_size) 
        
    def forward(self, x): 
        
        x = self.embedding(x) + self.pos_encoding(x) 
        
        for block in self.transformer_blocks: x = block(x) return self.lm_head(x) 