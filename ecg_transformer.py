
"""
Implementação do modelo ECG Transformer padrão
"""

import torch
import torch.nn as nn
import math

from .base_model import BaseModel
from ..config.model_configs import ECGTransformerConfig


class PositionalEncoding(nn.Module):
    """Positional Encoding para sequências"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ECGTransformer(BaseModel):
    """ECG Transformer para classificação de ECG"""
    
    def __init__(
        self,
        config: ECGTransformerConfig,
        num_classes: int = 5,
        input_channels: int = 12,
        **kwargs
    ):
        super().__init__(num_classes, input_channels)
        self.config = config
        
        # Linear projection from input_channels to d_model
        self.input_projection = nn.Linear(input_channels, config.d_model)
        
        self.pos_encoder = PositionalEncoding(config.d_model, config.dropout, config.max_seq_length)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, config.num_layers)
        
        self.fc_out = nn.Linear(config.d_model, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, input_channels, signal_length)
        # Transformer espera (batch_size, sequence_length, features)
        
        # Transpõe para (batch_size, signal_length, input_channels)
        x = x.permute(0, 2, 1)
        
        # Projeta para d_model
        x = self.input_projection(x)
        
        # Adiciona positional encoding
        x = self.pos_encoder(x)
        
        # Passa pelo Transformer Encoder
        transformer_output = self.transformer_encoder(x)
        
        # Agrega a saída do transformer (e.g., média sobre a dimensão da sequência)
        # Ou pode usar um token CLS se implementado
        aggregated_output = transformer_output.mean(dim=1) # Média sobre a sequência
        
        # Classificador
        output = self.fc_out(aggregated_output)
        
        return output


