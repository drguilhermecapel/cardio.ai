"""
Serviço de Explicabilidade para CardioAI
Implementa Grad-CAM, SHAP e outras técnicas de interpretabilidade
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import tensorflow as tf
from tensorflow import keras
import torch
import torch.nn.functional as F
from datetime import datetime
import io
import base64
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP não disponível. Instale com: pip install shap")


class ExplainabilityService:
    """Serviço para explicabilidade de modelos de ECG."""
    
    def __init__(self):
        self.explainers = {}
        
    def generate_gradcam(self, model, ecg_data: np.ndarray, 
                        target_class: Optional[int] = None) -> Dict[str, Any]:
        """Gera mapa de ativação Grad-CAM para modelo Keras."""
        try:
            if not isinstance(model, keras.Model):
                raise ValueError("Grad-CAM suporta apenas modelos Keras")
            
            # Preparar dados
            if ecg_data.ndim == 1:
                ecg_data = ecg_data.reshape(1, -1, 1)
            elif ecg_data.ndim == 2:
                ecg_data = ecg_data.reshape(ecg_data.shape[0], -1, 1)
            
            # Encontrar última camada convolucional
            last_conv_layer = None
            for layer in reversed(model.layers):
                if 'conv' in layer.name.lower():
                    last_conv_layer = layer
                    break
            
            if last_conv_layer is None:
                raise ValueError("Nenhuma camada convolucional encontrada")
            
            # Criar modelo para extração de features
            grad_model = keras.Model(
                inputs=model.input,
                outputs=[last_conv_layer.output, model.output]
            )
            
            # Calcular gradientes
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(ecg_data)
                
                if target_class is None:
                    target_class = tf.argmax(predictions[0])
                
                class_output = predictions[:, target_class]
            
            # Gradientes da classe em relação às features
            grads = tape.gradient(class_output, conv_outputs)
            
            # Pooling dos gradientes
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
            
            # Multiplicar features pelos gradientes
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
            
            # Normalizar heatmap
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            
            # Converter para numpy
            heatmap_np = heatmap.numpy()
            
            # Redimensionar para tamanho original do ECG
            original_length = ecg_data.shape[1]
            heatmap_resized = np.interp(
                np.linspace(0, len(heatmap_np)-1, original_length),
                np.arange(len(heatmap_np)),
                heatmap_np
            )
            
            # Gerar visualização
            visualization = self._create_gradcam_visualization(
                ecg_data[0, :, 0], heatmap_resized
            )
            
            return {
                "heatmap": heatmap_resized.tolist(),
                "target_class": int(target_class),
                "visualization": visualization,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro no Grad-CAM: {str(e)}")
            return {"error": str(e)}
    
    def generate_shap_explanation(self, model, ecg_data: np.ndarray, 
                                 background_data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Gera explicação SHAP para o modelo."""
        try:
            if not SHAP_AVAILABLE:
                raise ImportError("SHAP não está disponível")
            
            # Preparar dados
            if ecg_data.ndim == 1:
                ecg_data = ecg_data.reshape(1, -1)
            
            # Criar explainer
            if background_data is None:
                # Usar dados sintéticos como background
                background_data = np.random.normal(0, 1, (100, ecg_data.shape[1]))
            
            # Função de predição wrapper
            def predict_fn(x):
                if hasattr(model, 'predict'):
                    return model.predict(x)
                else:
                    # PyTorch
                    with torch.no_grad():
                        tensor_x = torch.FloatTensor(x)
                        return model(tensor_x).numpy()
            
            # Criar explainer
            explainer = shap.Explainer(predict_fn, background_data)
            
            # Calcular valores SHAP
            shap_values = explainer(ecg_data)
            
            # Gerar visualização
            visualization = self._create_shap_visualization(
                ecg_data[0], shap_values.values[0]
            )
            
            return {
                "shap_values": shap_values.values.tolist(),
                "base_values": shap_values.base_values.tolist(),
                "data": ecg_data.tolist(),
                "visualization": visualization,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro no SHAP: {str(e)}")
            return {"error": str(e)}
    
    def generate_feature_importance(self, model, ecg_data: np.ndarray) -> Dict[str, Any]:
        """Gera análise de importância de características."""
        try:
            # Análise de sensibilidade por perturbação
            baseline_pred = self._get_prediction(model, ecg_data)
            
            importance_scores = []
            window_size = max(1, len(ecg_data) // 50)  # 50 janelas
            
            for i in range(0, len(ecg_data), window_size):
                # Criar versão perturbada
                perturbed = ecg_data.copy()
                end_idx = min(i + window_size, len(ecg_data))
                perturbed[i:end_idx] = np.mean(ecg_data)  # Substituir por média
                
                # Calcular diferença na predição
                perturbed_pred = self._get_prediction(model, perturbed)
                importance = np.abs(baseline_pred - perturbed_pred)
                importance_scores.append(float(importance))
            
            # Criar visualização
            visualization = self._create_importance_visualization(
                ecg_data, importance_scores, window_size
            )
            
            return {
                "importance_scores": importance_scores,
                "window_size": window_size,
                "baseline_prediction": float(baseline_pred),
                "visualization": visualization,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de importância: {str(e)}")
            return {"error": str(e)}
    
    def _get_prediction(self, model, ecg_data: np.ndarray) -> float:
        """Obtém predição do modelo."""
        if ecg_data.ndim == 1:
            ecg_data = ecg_data.reshape(1, -1)
        
        if hasattr(model, 'predict'):
            pred = model.predict(ecg_data)
        else:
            with torch.no_grad():
                tensor_data = torch.FloatTensor(ecg_data)
                pred = model(tensor_data).numpy()
        
        return float(np.max(pred))
    
    def _create_gradcam_visualization(self, ecg_signal: np.ndarray, 
                                    heatmap: np.ndarray) -> str:
        """Cria visualização do Grad-CAM."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Sinal ECG original
        time_axis = np.arange(len(ecg_signal))
        ax1.plot(time_axis, ecg_signal, 'b-', linewidth=1)
        ax1.set_title('Sinal ECG Original')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # Heatmap sobreposto
        im = ax2.imshow(heatmap.reshape(1, -1), cmap='jet', alpha=0.7, 
                       aspect='auto', extent=[0, len(ecg_signal), -1, 1])
        ax2.plot(time_axis, ecg_signal / np.max(np.abs(ecg_signal)), 'k-', linewidth=2)
        ax2.set_title('Grad-CAM: Regiões Importantes')
        ax2.set_xlabel('Amostras')
        ax2.set_ylabel('Amplitude Normalizada')
        
        # Colorbar
        plt.colorbar(im, ax=ax2, label='Importância')
        
        plt.tight_layout()
        
        # Converter para base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def _create_shap_visualization(self, ecg_signal: np.ndarray, 
                                  shap_values: np.ndarray) -> str:
        """Cria visualização SHAP."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        time_axis = np.arange(len(ecg_signal))
        
        # Sinal original
        ax1.plot(time_axis, ecg_signal, 'b-', linewidth=1)
        ax1.set_title('Sinal ECG Original')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # Valores SHAP
        colors = ['red' if x > 0 else 'blue' for x in shap_values]
        ax2.bar(time_axis, shap_values, color=colors, alpha=0.7, width=1)
        ax2.set_title('Valores SHAP (Contribuição para Predição)')
        ax2.set_ylabel('Valor SHAP')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Combinado
        ax3.plot(time_axis, ecg_signal, 'k-', linewidth=1, alpha=0.7, label='ECG')
        ax3_twin = ax3.twinx()
        ax3_twin.fill_between(time_axis, 0, shap_values, 
                             color='red', alpha=0.3, label='SHAP')
        ax3.set_title('ECG + Contribuições SHAP')
        ax3.set_xlabel('Amostras')
        ax3.set_ylabel('Amplitude ECG')
        ax3_twin.set_ylabel('Valor SHAP')
        ax3.legend(loc='upper left')
        ax3_twin.legend(loc='upper right')
        
        plt.tight_layout()
        
        # Converter para base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def _create_importance_visualization(self, ecg_signal: np.ndarray, 
                                       importance_scores: List[float], 
                                       window_size: int) -> str:
        """Cria visualização de importância de características."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        time_axis = np.arange(len(ecg_signal))
        
        # Sinal ECG
        ax1.plot(time_axis, ecg_signal, 'b-', linewidth=1)
        ax1.set_title('Sinal ECG Original')
        ax1.set_ylabel('Amplitude')
        ax1.grid(True, alpha=0.3)
        
        # Importância por janelas
        window_centers = np.arange(0, len(ecg_signal), window_size) + window_size // 2
        window_centers = window_centers[:len(importance_scores)]
        
        ax2.bar(window_centers, importance_scores, width=window_size, 
               alpha=0.7, color='orange')
        ax2.set_title('Importância por Região (Análise de Sensibilidade)')
        ax2.set_xlabel('Amostras')
        ax2.set_ylabel('Score de Importância')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Converter para base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64
    
    def generate_comprehensive_report(self, model, ecg_data: np.ndarray, 
                                    model_name: str) -> Dict[str, Any]:
        """Gera relatório completo de explicabilidade."""
        try:
            report = {
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
                "ecg_length": len(ecg_data),
                "analyses": {}
            }
            
            # Grad-CAM (se aplicável)
            if hasattr(model, 'layers'):
                gradcam_result = self.generate_gradcam(model, ecg_data)
                if "error" not in gradcam_result:
                    report["analyses"]["gradcam"] = gradcam_result
            
            # SHAP (se disponível)
            if SHAP_AVAILABLE:
                shap_result = self.generate_shap_explanation(model, ecg_data)
                if "error" not in shap_result:
                    report["analyses"]["shap"] = shap_result
            
            # Análise de importância
            importance_result = self.generate_feature_importance(model, ecg_data)
            if "error" not in importance_result:
                report["analyses"]["feature_importance"] = importance_result
            
            return report
            
        except Exception as e:
            logger.error(f"Erro no relatório de explicabilidade: {str(e)}")
            return {"error": str(e)}


# Instância global do serviço
explainability_service = ExplainabilityService()

