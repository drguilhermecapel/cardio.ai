
"""
Implementação do modelo híbrido CNN-LSTM para classificação de ECG
"""

import torch
import torch.nn as nn

from .base_model import BaseModel
from ..config.model_configs import CNNLSTMConfig


class CNNLSTM(BaseModel):
    """Modelo CNN-LSTM para classificação de ECG"""
    
    def __init__(
        self,
        config: CNNLSTMConfig,
        num_classes: int = 5,
        input_channels: int = 12,
        **kwargs
    ):
        super().__init__(num_classes, input_channels)
        self.config = config
        
        # CNN Feature Extractor
        cnn_layers = []
        in_c = input_channels
        for i, (out_c, k_size) in enumerate(zip(config.cnn_channels, config.kernel_sizes)):
            cnn_layers.append(nn.Conv1d(in_c, out_c, kernel_size=k_size, padding=k_size//2))
            cnn_layers.append(nn.BatchNorm1d(out_c))
            cnn_layers.append(nn.ReLU())
            cnn_layers.append(nn.MaxPool1d(kernel_size=2, stride=2)) # Reduz dimensão pela metade
            in_c = out_c
        self.cnn = nn.Sequential(*cnn_layers)
        
        # LSTM Layer
        # Precisamos calcular a dimensão de entrada para a LSTM
        # Isso depende do comprimento do sinal após a CNN e do número de canais finais da CNN
        # Para simplificar, vamos usar um Linear para mapear para lstm_hidden_size
        # Ou, podemos passar o output da CNN para a LSTM diretamente se for 1D
        
        # Dummy forward para calcular a dimensão de entrada da LSTM
        with torch.no_grad():
            dummy_input = torch.randn(1, input_channels, 5000) # Assumindo 5000 como comprimento padrão
            cnn_output = self.cnn(dummy_input)
            lstm_input_dim = cnn_output.shape[1] * cnn_output.shape[2] # Flatten
            
        self.lstm = nn.LSTM(
            input_size=in_c, # O número de canais da última camada CNN
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_layers,
            batch_first=True,
            bidirectional=config.bidirectional
        )
        
        # Classifier head
        lstm_output_dim = config.lstm_hidden_size * (2 if config.bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(lstm_output_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, input_channels, signal_length)
        
        # CNN Feature Extraction
        cnn_output = self.cnn(x) # (batch_size, last_cnn_channels, reduced_length)
        
        # Permute para (batch_size, reduced_length, last_cnn_channels) para LSTM
        cnn_output = cnn_output.permute(0, 2, 1) 
        
        # LSTM
        lstm_output, _ = self.lstm(cnn_output)
        
        # Usar o último hidden state da LSTM para classificação
        # Se bidirecional, concatenar os últimos hidden states forward e backward
        if self.config.bidirectional:
            # lstm_output contém (batch, seq_len, num_directions * hidden_size)
            # Pegamos o último para forward e o primeiro para backward
            forward_output = lstm_output[:, -1, :self.config.lstm_hidden_size]
            backward_output = lstm_output[:, 0, self.config.lstm_hidden_size:]
            lstm_final_output = torch.cat((forward_output, backward_output), dim=1)
        else:
            lstm_final_output = lstm_output[:, -1, :]
            
        # Classifier
        output = self.classifier(lstm_final_output)
        return output


