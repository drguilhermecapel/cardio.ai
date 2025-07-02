
"""
Fábrica para criação de instâncias de modelos de ECG
"""

from typing import Dict, Any

from .base_model import BaseModel
from .heartbeit import HeartBEiT
from .cnn_lstm import CNNLSTM
from .se_resnet1d import SEResNet1D
from .ecg_transformer import ECGTransformer
from ..config.model_configs import get_model_config, ModelConfig


class ModelFactory:
    """Classe para criar instâncias de modelos de deep learning de ECG"""
    
    @staticmethod
    def create_model(
        model_name: str,
        num_classes: int,
        input_channels: int,
        pretrained_path: str = None,
        **kwargs
    ) -> BaseModel:
        """
        Cria e retorna uma instância de um modelo de ECG.
        
        Args:
            model_name: Nome do modelo (ex: "heartbeit", "cnn_lstm", "se_resnet1d", "ecg_transformer")
            num_classes: Número de classes para a camada de saída do modelo.
            input_channels: Número de canais de entrada (derivações do ECG).
            pretrained_path: Caminho para pesos pré-treinados (opcional).
            **kwargs: Argumentos adicionais a serem passados para o construtor do modelo.
            
        Returns:
            Uma instância de BaseModel ou uma de suas subclasses.
            
        Raises:
            ValueError: Se o model_name não for reconhecido.
        """
        
        model_name = model_name.lower()
        
        # Obter a configuração específica do modelo
        model_config: ModelConfig = get_model_config(model_name, num_classes=num_classes, input_channels=input_channels, **kwargs)
        
        if model_name == "heartbeit":
            model = HeartBEiT(config=model_config, num_classes=num_classes, in_chans=input_channels, **kwargs)
        elif model_name == "cnn_lstm":
            model = CNNLSTM(config=model_config, num_classes=num_classes, input_channels=input_channels, **kwargs)
        elif model_name == "se_resnet1d":
            model = SEResNet1D(config=model_config, num_classes=num_classes, input_channels=input_channels, **kwargs)
        elif model_name == "ecg_transformer":
            model = ECGTransformer(config=model_config, num_classes=num_classes, input_channels=input_channels, **kwargs)
        else:
            raise ValueError(f"Modelo {model_name} não suportado. "
                            f"Modelos disponíveis: heartbeit, cnn_lstm, se_resnet1d, ecg_transformer")
                            
        if pretrained_path:
            model.load_pretrained(pretrained_path)
            
        return model


