
"""
Configurações específicas para cada arquitetura de modelo
"""

from typing import Dict, Any
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Configuração base para modelos"""
    name: str
    num_classes: int = 5  # Padrão para 5 superclasses do PTB-XL
    input_channels: int = 12  # 12 derivações
    pretrained_path: str = None
    
    
@dataclass
class HeartBEiTConfig(ModelConfig):
    """Configuração para HeartBEiT transformer"""
    name: str = "heartbeit"
    patch_size: int = 20
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    qkv_bias: bool = True
    drop_rate: float = 0.1
    attn_drop_rate: float = 0.1
    drop_path_rate: float = 0.1
    use_abs_pos_emb: bool = True
    masked_ratio: float = 0.75  # Para pré-treinamento
    

@dataclass 
class CNNLSTMConfig(ModelConfig):
    """Configuração para modelo híbrido CNN-LSTM"""
    name: str = "cnn_lstm"
    cnn_channels: list = field(default_factory=lambda: [64, 128, 256, 512])
    kernel_sizes: list = field(default_factory=lambda: [7, 5, 3, 3])
    lstm_hidden_size: int = 256
    lstm_layers: int = 2
    bidirectional: bool = True
    dropout: float = 0.3
    

@dataclass
class SEResNet1DConfig(ModelConfig):
    """Configuração para SE-ResNet1D"""
    name: str = "se_resnet1d"
    layers: list = field(default_factory=lambda: [3, 4, 6, 3])
    channels: list = field(default_factory=lambda: [64, 128, 256, 512])
    kernel_size: int = 7
    reduction: int = 16  # Para SE blocks
    dropout: float = 0.2
    

@dataclass
class ECGTransformerConfig(ModelConfig):
    """Configuração para ECG Transformer padrão"""
    name: str = "ecg_transformer"
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    activation: str = "gelu"
    max_seq_length: int = 5000
    

# Registry de configurações
MODEL_CONFIGS = {
    "heartbeit": HeartBEiTConfig,
    "cnn_lstm": CNNLSTMConfig,
    "se_resnet1d": SEResNet1DConfig,
    "ecg_transformer": ECGTransformerConfig,
}


def get_model_config(model_name: str, **kwargs) -> ModelConfig:
    """Retorna configuração do modelo com parâmetros customizados"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Modelo {model_name} não encontrado. "
                        f"Disponíveis: {list(MODEL_CONFIGS.keys())}")
    
    config_class = MODEL_CONFIGS[model_name]
    return config_class(**kwargs)


