from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer

import torch
import torch.nn as nn
import math
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer


class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the div term: 10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but should be saved/loaded)
        self.register_buffer('pe', pe)
    
    def forward(self, x):

        # Add positional encoding to input
        x = x + self.pe[:, :x.size(1), :]
        
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(LearnablePositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))
    
    def forward(self, x):

        x = x + self.pos_embedding[:, :x.size(1), :]
        
        return self.dropout(x)


class CausalAttention(nn.Module):
    def __init__(
        self,
        embed_size=1024,
        n_heads=4,
        num_layers=2,
        use_positional_encoding=True,
        positional_encoding_type='sinusoidal',
        max_len=5000,
        pe_dropout=0.1
    ):
        super(CausalAttention, self).__init__()
        
        self.use_positional_encoding = use_positional_encoding
        
        # Add positional encoding
        if use_positional_encoding:
            if positional_encoding_type == 'sinusoidal':
                self.pos_encoder = PositionalEncoding(
                    d_model=embed_size,
                    max_len=max_len,
                    dropout=pe_dropout
                )
            elif positional_encoding_type == 'learnable':
                self.pos_encoder = LearnablePositionalEncoding(
                    d_model=embed_size,
                    max_len=max_len,
                    dropout=pe_dropout
                )
            else:
                raise ValueError(f"Unknown positional_encoding_type: {positional_encoding_type}")

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            mask_flag=True,  # Enable causal masking
                            factor=1,
                            attention_dropout=0.1,
                            output_attention=True,
                        ),
                        embed_size,
                        n_heads,
                    ),
                    embed_size,
                    256,
                    dropout=0.01,
                    activation="gelu",
                )
                for l in range(num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(embed_size),
        )

    def forward(self, combined, padding_mask=None):

        # Add positional encoding
        if self.use_positional_encoding:
            combined = self.pos_encoder(combined)
        
        # Apply causal attention
        # Note: The encoder should handle padding_mask if provided
        if padding_mask is not None:
            output, attention = self.encoder(combined, attn_mask=padding_mask)
        else:
            output, attention = self.encoder(combined)

        return output, attention
