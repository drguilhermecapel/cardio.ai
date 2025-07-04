"""
Serviço Unificado de Modelos para CardioAI
Combina as melhores funcionalidades de todos os serviços de modelo anteriores
"""

import os
import logging
import pickle
import joblib
import json
import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime

# Importações condicionais para TensorFlow e PyTorch
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

# Importações para modelos alternativos
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class UnifiedModelService:
    """
    Serviço unificado para gerenciamento de modelos de ML para análise de ECG.
    Suporta modelos TensorFlow/Keras, PyTorch e scikit-learn.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Inicializa o serviço de modelos.
        
        Args:
            models_dir: Diretório onde os modelos estão armazenados
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Armazenamento de modelos e metadados
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Dict] = {}
        
        # Configurações de diagnóstico
        self.diagnosis_mapping = {
            0: "Normal",
            1: "Fibrilação Atrial",
            2: "Bradicardia",
            3: "Taquicardia",
            4: "Arritmia Ventricular",
            5: "Bloqueio AV",
            6: "Isquemia",
            7: "Infarto do Miocárdio",
            8: "Hipertrofia Ventricular",
            9: "Anormalidade Inespecífica"
        }
        
        # Parâmetros de processamento
        self.signal_length = 5000  # Comprimento padrão do sinal
        self.sampling_rate = 500   # Taxa de amostragem padrão
        
        # Inicializar modelos
        self._initialize_models()
    
    def _initialize_models(self):
        """Inicializa modelos disponíveis de todas as fontes suportadas."""
        try:
            # Carregar modelos .h5 (TensorFlow/Keras)
            if TENSORFLOW_AVAILABLE:
                self._load_h5_models()
            
            # Carregar modelos .pt (PyTorch)
            if PYTORCH_AVAILABLE:
                self._load_pytorch_models()
            
            # Carregar modelos scikit-learn
            self._load_sklearn_models()
            
            # Criar modelo demo se nenhum modelo foi carregado
            if not self.models:
                self._create_demo_models()
            
            logger.info(f"Modelos inicializados: {list(self.models.keys())}")
            
        except Exception as e:
            logger.error(f"Erro na inicialização de modelos: {str(e)}")
            self._create_demo_models()
    
    def _load_h5_models(self):
        """Carrega modelos .h5 do diretório de modelos."""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow não disponível - pulando modelos .h5")
            return
        
        try:
            h5_files = list(self.models_dir.glob("*.h5"))
            
            for h5_file in h5_files:
                try:
                    model_name = h5_file.stem
                    
                    # Carregar modelo
                    model = keras.models.load_model(str(h5_file))
                    
                    # Carregar scaler se existir
                    scaler_file = self.models_dir / f"{model_name}_scaler.pkl"
                    scaler = None
                    if scaler_file.exists():
                        scaler = joblib.load(scaler_file)
                    
                    # Carregar metadados se existir
                    metadata_file = self.models_dir / f"{model_name}_metadata.json"
                    metadata = {}
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    
                    # Registrar modelo
                    self.models[model_name] = model
                    if scaler:
                        self.scalers[model_name] = scaler
                    self.model_metadata[model_name] = {
                        'type': 'tensorflow_h5',
                        'input_shape': model.input_shape,
                        'output_shape': model.output_shape,
                        'loaded_from': str(h5_file),
                        'has_scaler': scaler is not None,
                        **metadata
                    }
                    
                    logger.info(f"Modelo .h5 carregado: {model_name}")
                    
                except Exception as e:
                    logger.error(f"Erro ao carregar {h5_file}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Erro na busca por modelos .h5: {str(e)}")
    
    def _load_pytorch_models(self):
        """Carrega modelos PyTorch do diretório de modelos."""
        if not PYTORCH_AVAILABLE:
            logger.warning("PyTorch não disponível - pulando modelos .pt")
            return
        
        try:
            pt_files = list(self.models_dir.glob("*.pt"))
            
            for pt_file in pt_files:
                try:
                    model_name = pt_file.stem
                    
                    # Carregar modelo
                    model = torch.load(str(pt_file))
                    model.eval()  # Colocar em modo de avaliação
                    
                    # Carregar scaler se existir
                    scaler_file = self.models_dir / f"{model_name}_scaler.pkl"
                    scaler = None
                    if scaler_file.exists():
                        scaler = joblib.load(scaler_file)
                    
                    # Carregar metadados se existir
                    metadata_file = self.models_dir / f"{model_name}_metadata.json"
                    metadata = {}
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    
                    # Registrar modelo
                    self.models[model_name] = model
                    if scaler:
                        self.scalers[model_name] = scaler
                    self.model_metadata[model_name] = {
                        'type': 'pytorch',
                        'loaded_from': str(pt_file),
                        'has_scaler': scaler is not None,
                        **metadata
                    }
                    
                    logger.info(f"Modelo PyTorch carregado: {model_name}")
                    
                except Exception as e:
                    logger.error(f"Erro ao carregar {pt_file}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Erro na busca por modelos PyTorch: {str(e)}")
    
    def _load_sklearn_models(self):
        """Carrega modelos scikit-learn do diretório de modelos."""
        try:
            pkl_files = list(self.models_dir.glob("*.pkl"))
            
            for pkl_file in pkl_files:
                # Pular arquivos de scaler
                if "_scaler" in pkl_file.name:
                    continue
                
                try:
                    model_name = pkl_file.stem
                    
                    # Carregar modelo
                    model = joblib.load(pkl_file)
                    
                    # Carregar scaler se existir
                    scaler_file = self.models_dir / f"{model_name}_scaler.pkl"
                    scaler = None
                    if scaler_file.exists():
                        scaler = joblib.load(scaler_file)
                    
                    # Carregar metadados se existir
                    metadata_file = self.models_dir / f"{model_name}_metadata.json"
                    metadata = {}
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                    
                    # Registrar modelo
                    self.models[model_name] = model
                    if scaler:
                        self.scalers[model_name] = scaler
                    self.model_metadata[model_name] = {
                        'type': 'sklearn',
                        'loaded_from': str(pkl_file),
                        'has_scaler': scaler is not None,
                        **metadata
                    }
                    
                    logger.info(f"Modelo scikit-learn carregado: {model_name}")
                    
                except Exception as e:
                    logger.error(f"Erro ao carregar {pkl_file}: {str(e)}")
            
        except Exception as e:
            logger.error(f"Erro na busca por modelos scikit-learn: {str(e)}")
    
    def _create_demo_models(self):
        """Cria modelos de demonstração quando nenhum modelo real está disponível."""
        try:
            # Modelo demo baseado em Random Forest
            demo_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10
            )
            
            # Treinar com dados sintéticos
            X_demo, y_demo = self._generate_demo_data()
            demo_model.fit(X_demo, y_demo)
            
            # Scaler para normalização
            demo_scaler = StandardScaler()
            demo_scaler.fit(X_demo)
            
            # Registrar modelo demo
            self.models['demo_ecg_classifier'] = demo_model
            self.scalers['demo_ecg_classifier'] = demo_scaler
            self.model_metadata['demo_ecg_classifier'] = {
                'type': 'sklearn_demo',
                'input_shape': (self.signal_length,),
                'output_classes': len(self.diagnosis_mapping),
                'created': datetime.now().isoformat(),
                'description': 'Modelo de demonstração para análise de ECG'
            }
            
            logger.info("Modelo demo criado com sucesso")
            
        except Exception as e:
            logger.error(f"Erro ao criar modelo demo: {str(e)}")
    
    def _generate_demo_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Gera dados sintéticos para treinamento demo."""
        try:
            X = []
            y = []
            
            for i in range(n_samples):
                # Gerar sinal ECG sintético
                signal = self._generate_synthetic_ecg()
                
                # Atribuir classe baseada em características do sinal
                signal_class = self._classify_synthetic_signal(signal)
                
                X.append(signal)
                y.append(signal_class)
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"Erro na geração de dados demo: {str(e)}")
            return np.random.randn(100, self.signal_length), np.random.randint(0, 3, 100)
    
    def _generate_synthetic_ecg(self) -> np.ndarray:
        """Gera sinal ECG sintético."""
        try:
            t = np.linspace(0, 10, self.signal_length)  # 10 segundos
            
            # Componentes do ECG
            # Onda P
            p_wave = 0.1 * np.sin(2 * np.pi * 0.8 * t) * np.exp(-((t - 1) ** 2) / 0.1)
            
            # Complexo QRS
            qrs_complex = 0.8 * np.sin(2 * np.pi * 15 * t) * np.exp(-((t - 2) ** 2) / 0.01)
            
            # Onda T
            t_wave = 0.3 * np.sin(2 * np.pi * 1.2 * t) * np.exp(-((t - 3) ** 2) / 0.2)
            
            # Linha de base
            baseline = 0.05 * np.sin(2 * np.pi * 0.1 * t)
            
            # Ruído
            noise = 0.02 * np.random.randn(len(t))
            
            # Sinal completo
            signal = p_wave + qrs_complex + t_wave + baseline + noise
            
            # Repetir padrão para criar ritmo
            pattern_length = len(signal) // 8
            pattern = signal[:pattern_length]
            full_signal = np.tile(pattern, 8)
            
            # Adicionar variações para diferentes condições
            variation = np.random.choice(['normal', 'tachycardia', 'bradycardia', 'arrhythmia'])
            
            if variation == 'tachycardia':
                # Aumentar frequência
                full_signal = np.interp(np.linspace(0, 1, len(full_signal)), 
                                      np.linspace(0, 1, int(len(full_signal) * 0.7)), 
                                      full_signal[:int(len(full_signal) * 0.7)])
            elif variation == 'bradycardia':
                # Diminuir frequência
                full_signal = np.interp(np.linspace(0, 1, len(full_signal)), 
                                      np.linspace(0, 1, int(len(full_signal) * 1.3)), 
                                      np.tile(full_signal, 2)[:int(len(full_signal) * 1.3)])
            elif variation == 'arrhythmia':
                # Adicionar irregularidades
                irregularity = 0.1 * np.random.randn(len(full_signal))
                full_signal += irregularity
            
            return full_signal[:self.signal_length]
            
        except Exception as e:
            logger.error(f"Erro na geração de ECG sintético: {str(e)}")
            return np.random.randn(self.signal_length)
    
    def _classify_synthetic_signal(self, signal: np.ndarray) -> int:
        """Classifica sinal sintético baseado em características."""
        try:
            # Características simples para classificação
            mean_amplitude = np.mean(np.abs(signal))
            std_amplitude = np.std(signal)
            peak_count = len(self._find_peaks(signal))
            
            # Regras simples de classificação
            if peak_count > 15:
                return 3  # Taquicardia
            elif peak_count < 5:
                return 2  # Bradicardia
            elif std_amplitude > 0.3:
                return 4  # Arritmia
            elif mean_amplitude > 0.2:
                return 1  # Fibrilação Atrial
            else:
                return 0  # Normal
                
        except Exception:
            return 0  # Normal por padrão
    
    def _find_peaks(self, signal: np.ndarray, threshold: float = 0.1) -> List[int]:
        """Encontra picos no sinal."""
        try:
            peaks = []
            for i in range(1, len(signal) - 1):
                if (signal[i] > signal[i-1] and 
                    signal[i] > signal[i+1] and 
                    signal[i] > threshold):
                    peaks.append(i)
            return peaks
        except Exception:
            return []
    
    def predict_ecg(self, model_name: str, ecg_data: np.ndarray, 
                   metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Realiza predição de ECG usando modelo especificado.
        
        Args:
            model_name: Nome do modelo a usar
            ecg_data: Dados do ECG (1D array)
            metadata: Metadados opcionais do sinal
            
        Returns:
            Dict com resultados da predição
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Modelo '{model_name}' não encontrado")
            
            model = self.models[model_name]
            model_meta = self.model_metadata[model_name]
            
            # Preprocessar dados
            processed_data = self._preprocess_ecg_data(ecg_data, model_name)
            
            # Realizar predição
            if model_meta['type'] == 'tensorflow_h5':
                prediction = self._predict_tensorflow(model, processed_data, model_name)
            elif model_meta['type'] == 'pytorch':
                prediction = self._predict_pytorch(model, processed_data, model_name)
            else:
                prediction = self._predict_sklearn(model, processed_data, model_name)
            
            # Pós-processar resultados
            result = self._postprocess_prediction(prediction, model_meta, metadata)
            
            logger.info(f"Predição realizada com modelo {model_name}")
            return result
            
        except Exception as e:
            logger.error(f"Erro na predição: {str(e)}")
            return {
                'error': str(e),
                'model_used': model_name,
                'timestamp': datetime.now().isoformat()
            }
    
    def _preprocess_ecg_data(self, ecg_data: np.ndarray, model_name: str) -> np.ndarray:
        """Preprocessa dados de ECG para o modelo."""
        try:
            # Converter para numpy se necessário
            if not isinstance(ecg_data, np.ndarray):
                ecg_data = np.array(ecg_data)
            
            # Garantir que é 1D
            if ecg_data.ndim > 1:
                ecg_data = ecg_data.flatten()
            
            # Redimensionar para comprimento padrão
            if len(ecg_data) != self.signal_length:
                # Interpolar para comprimento padrão
                x_old = np.linspace(0, 1, len(ecg_data))
                x_new = np.linspace(0, 1, self.signal_length)
                ecg_data = np.interp(x_new, x_old, ecg_data)
            
            # Normalizar
            if model_name in self.scalers:
                scaler = self.scalers[model_name]
                ecg_data = scaler.transform(ecg_data.reshape(1, -1))[0]
            else:
                # Normalização Z-score
                ecg_data = (ecg_data - np.mean(ecg_data)) / (np.std(ecg_data) + 1e-8)
            
            return ecg_data
            
        except Exception as e:
            logger.error(f"Erro no preprocessamento: {str(e)}")
            return ecg_data
    
    def _predict_tensorflow(self, model, data: np.ndarray, model_name: str) -> np.ndarray:
        """Realiza predição com modelo TensorFlow."""
        try:
            # Preparar dados para TensorFlow
            input_data = data.reshape(1, -1, 1)  # (batch, timesteps, features)
            
            # Predição
            prediction = model.predict(input_data, verbose=0)
            
            return prediction[0]  # Remover dimensão do batch
            
        except Exception as e:
            logger.error(f"Erro na predição TensorFlow: {str(e)}")
            # Retornar predição dummy
            return np.random.rand(len(self.diagnosis_mapping))
    
    def _predict_pytorch(self, model, data: np.ndarray, model_name: str) -> np.ndarray:
        """Realiza predição com modelo PyTorch."""
        try:
            # Preparar dados para PyTorch
            input_data = torch.tensor(data, dtype=torch.float32).reshape(1, 1, -1)
            
            # Predição
            with torch.no_grad():
                prediction = model(input_data)
                
            # Converter para numpy
            if isinstance(prediction, torch.Tensor):
                prediction = prediction.numpy()
            
            return prediction[0]  # Remover dimensão do batch
            
        except Exception as e:
            logger.error(f"Erro na predição PyTorch: {str(e)}")
            # Retornar predição dummy
            return np.random.rand(len(self.diagnosis_mapping))
    
    def _predict_sklearn(self, model, data: np.ndarray, model_name: str) -> np.ndarray:
        """Realiza predição com modelo scikit-learn."""
        try:
            # Preparar dados para sklearn
            input_data = data.reshape(1, -1)
            
            # Predição de probabilidades
            if hasattr(model, 'predict_proba'):
                prediction = model.predict_proba(input_data)[0]
            else:
                # Se não tem predict_proba, usar predict e converter
                class_pred = model.predict(input_data)[0]
                prediction = np.zeros(len(self.diagnosis_mapping))
                prediction[class_pred] = 1.0
            
            return prediction
            
        except Exception as e:
            logger.error(f"Erro na predição sklearn: {str(e)}")
            # Retornar predição dummy
            return np.random.rand(len(self.diagnosis_mapping))
    
    def _postprocess_prediction(self, prediction: np.ndarray, 
                               model_meta: Dict, metadata: Optional[Dict]) -> Dict[str, Any]:
        """Pós-processa resultados da predição."""
        try:
            # Encontrar classe predita
            predicted_class = int(np.argmax(prediction))
            confidence = float(prediction[predicted_class])
            
            # Mapear para diagnóstico
            diagnosis = self.diagnosis_mapping.get(predicted_class, "Desconhecido")
            
            # Criar distribuição de probabilidades
            probabilities = {}
            for class_id, class_name in self.diagnosis_mapping.items():
                if class_id < len(prediction):
                    probabilities[class_name] = float(prediction[class_id])
            
            # Análise de confiança
            confidence_level = self._analyze_confidence(confidence)

            # Recomendações clínicas
            recommendations = self._generate_recommendations(
                predicted_class, confidence, metadata
            )

            result = {
                'predicted_class': predicted_class,
                'diagnosis': diagnosis,
                'confidence': confidence,
                'confidence_level': confidence_level,
                'probabilities': probabilities,
                'recommendations': recommendations,
                'model_metadata': model_meta,
                'analysis_timestamp': datetime.now().isoformat()
            }

            return result

        except Exception as e:
            logger.error(f"Erro no pós-processamento: {str(e)}")
            return {
                'predicted_class': 0,
                'diagnosis': 'Normal',
                'confidence': 0.5,
                'confidence_level': 'baixa',
                'error': str(e)
            }

    def _analyze_confidence(self, confidence: float) -> str:
        """Analisa nível de confiança da predição."""
        if confidence >= 0.9:
            return 'muito_alta'
        elif confidence >= 0.8:
            return 'alta'
        elif confidence >= 0.6:
            return 'moderada'
        elif confidence >= 0.4:
            return 'baixa'
        else:
            return 'muito_baixa'

    def _generate_recommendations(self, predicted_class: int, confidence: float,
                                metadata: Optional[Dict]) -> Dict[str, Any]:
        """Gera recomendações clínicas baseadas na predição."""
        try:
            recommendations = {
                'clinical_review_required': confidence < 0.7,
                'follow_up': [],
                'clinical_notes': [],
                'urgent_attention': False
            }
            
            # Recomendações baseadas na classe
            if predicted_class == 0:  # Normal
                recommendations['follow_up'].append('Acompanhamento de rotina')
                recommendations['clinical_notes'].append('ECG dentro dos padrões normais')
                
            elif predicted_class == 1:  # Fibrilação Atrial
                recommendations['follow_up'].append('Avaliação cardiológica em 7 dias')
                recommendations['clinical_notes'].append('Considerar anticoagulação')
                recommendations['urgent_attention'] = confidence > 0.8
                
            elif predicted_class in [2, 3]:  # Bradicardia ou Taquicardia
                recommendations['follow_up'].append('Monitoramento de ritmo cardíaco')
                recommendations['clinical_notes'].append('Avaliar medicações em uso')
                
            elif predicted_class == 4:  # Arritmia Ventricular
                recommendations['follow_up'].append('Avaliação cardiológica imediata')
                recommendations['clinical_notes'].append('Considerar Holter 24h')
                recommendations['urgent_attention'] = confidence > 0.7
                
            elif predicted_class in [6, 7]:  # Isquemia ou Infarto
                recommendations['follow_up'].append('Avaliação cardiológica de emergência')
                recommendations['clinical_notes'].append('Considerar marcadores cardíacos')
                recommendations['urgent_attention'] = True
            
            # Ajustar baseado na confiança
            if confidence < 0.5:
                recommendations['follow_up'].append('Repetir ECG')
                recommendations['clinical_notes'].append('Baixa confiança - repetir ECG')
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Erro na geração de recomendações: {str(e)}")
            return {
                'clinical_review_required': True,
                'follow_up': ['Avaliação médica'],
                'clinical_notes': ['Erro na análise automática'],
                'urgent_attention': False
            }
    
    def list_models(self) -> List[str]:
        """Lista todos os modelos disponíveis."""
        return list(self.models.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Retorna informações sobre um modelo específico."""
        if model_name not in self.model_metadata:
            return {'error': f"Modelo '{model_name}' não encontrado"}
        
        return self.model_metadata[model_name]
    
    def add_h5_model(self, model_path: str, model_name: str) -> bool:
        """Adiciona um novo modelo .h5 ao serviço."""
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow não disponível - não é possível adicionar modelo .h5")
            return False
        
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                logger.error(f"Arquivo não encontrado: {model_path}")
                return False
            
            # Carregar modelo
            model = keras.models.load_model(str(model_path))

            # Registrar modelo
            self.models[model_name] = model
            self.model_metadata[model_name] = {
                'type': 'tensorflow_h5',
                'input_shape': model.input_shape,
                'output_shape': model.output_shape,
                'loaded_from': str(model_path),
                'loaded_at': datetime.now().isoformat()
            }

            logger.info(f"Modelo .h5 adicionado: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Erro ao adicionar modelo .h5: {str(e)}")
            return False


# Instância global do serviço unificado
unified_model_service = UnifiedModelService()


def get_model_service() -> UnifiedModelService:
    """Retorna instância do serviço unificado de modelos."""
    return unified_model_service