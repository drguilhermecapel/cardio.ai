"""
Script para exportar modelos treinados para produção
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging
import argparse
import json
from typing import Dict, Any

from backend.training.config.training_config import training_config
from backend.training.models.model_factory import ModelFactory
from backend.training.config.model_configs import get_model_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelExporter:
    """Classe para exportar modelos treinados para produção"""
    
    def __init__(self, export_root: Path = None):
        self.export_root = export_root or training_config.EXPORT_ROOT
        self.export_root.mkdir(parents=True, exist_ok=True)
        
    def export_model(
        self,
        model_name: str,
        checkpoint_path: str,
        num_classes: int,
        input_channels: int = 12,
        export_format: str = "pytorch",
        model_info: Dict[str, Any] = None
    ):
        """Exporta um modelo treinado para produção"""
        
        logger.info(f"Exportando modelo {model_name} de {checkpoint_path}")
        
        # Carregar modelo
        model = ModelFactory.create_model(
            model_name=model_name,
            num_classes=num_classes,
            input_channels=input_channels
        )
        
        # Carregar pesos do checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        
        # Criar diretório de exportação
        export_dir = self.export_root / f"{model_name}_exported"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        if export_format == "pytorch":
            self._export_pytorch(model, export_dir, model_info)
        elif export_format == "onnx":
            self._export_onnx(model, export_dir, input_channels, model_info)
        elif export_format == "torchscript":
            self._export_torchscript(model, export_dir, input_channels, model_info)
        else:
            raise ValueError(f"Formato de exportação {export_format} não suportado")
            
        logger.info(f"Modelo exportado para {export_dir}")
        return export_dir
        
    def _export_pytorch(self, model: nn.Module, export_dir: Path, model_info: Dict = None):
        """Exporta modelo no formato PyTorch padrão"""
        model_path = export_dir / "model.pth"
        torch.save(model.state_dict(), model_path)
        
        # Salvar informações do modelo
        info = {
            "model_class": model.__class__.__name__,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "input_shape": [12, 5000],  # [channels, length]
            "output_shape": [model.num_classes],
            "framework": "pytorch",
            **(model_info or {})
        }
        
        info_path = export_dir / "model_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
            
        logger.info(f"Modelo PyTorch salvo em {model_path}")
        
    def _export_onnx(self, model: nn.Module, export_dir: Path, input_channels: int, model_info: Dict = None):
        """Exporta modelo no formato ONNX"""
        try:
            import onnx
        except ImportError:
            logger.error("ONNX não está instalado. Execute: pip install onnx")
            return
            
        model_path = export_dir / "model.onnx"
        
        # Criar input dummy
        dummy_input = torch.randn(1, input_channels, 5000)
        
        # Exportar para ONNX
        torch.onnx.export(
            model,
            dummy_input,
            model_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Salvar informações do modelo
        info = {
            "model_class": model.__class__.__name__,
            "input_shape": [input_channels, 5000],
            "output_shape": [model.num_classes],
            "framework": "onnx",
            "opset_version": 11,
            **(model_info or {})
        }
        
        info_path = export_dir / "model_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
            
        logger.info(f"Modelo ONNX salvo em {model_path}")
        
    def _export_torchscript(self, model: nn.Module, export_dir: Path, input_channels: int, model_info: Dict = None):
        """Exporta modelo no formato TorchScript"""
        model_path = export_dir / "model.pt"
        
        # Criar input dummy
        dummy_input = torch.randn(1, input_channels, 5000)
        
        # Converter para TorchScript
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(model_path)
        
        # Salvar informações do modelo
        info = {
            "model_class": model.__class__.__name__,
            "input_shape": [input_channels, 5000],
            "output_shape": [model.num_classes],
            "framework": "torchscript",
            **(model_info or {})
        }
        
        info_path = export_dir / "model_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
            
        logger.info(f"Modelo TorchScript salvo em {model_path}")


def main():
    parser = argparse.ArgumentParser(description="Exportar modelos treinados")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Nome do modelo")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Caminho para o checkpoint do modelo")
    parser.add_argument("--num_classes", type=int, required=True,
                        help="Número de classes do modelo")
    parser.add_argument("--input_channels", type=int, default=12,
                        help="Número de canais de entrada")
    parser.add_argument("--format", type=str, default="pytorch",
                        choices=["pytorch", "onnx", "torchscript"],
                        help="Formato de exportação")
    parser.add_argument("--export_root", type=str, default=None,
                        help="Diretório raiz para exportação")
    
    args = parser.parse_args()
    
    export_root = Path(args.export_root) if args.export_root else None
    exporter = ModelExporter(export_root)
    
    try:
        export_dir = exporter.export_model(
            model_name=args.model_name,
            checkpoint_path=args.checkpoint_path,
            num_classes=args.num_classes,
            input_channels=args.input_channels,
            export_format=args.format
        )
        logger.info(f"Exportação concluída: {export_dir}")
    except Exception as e:
        logger.error(f"Erro durante a exportação: {e}")


if __name__ == "__main__":
    main()

