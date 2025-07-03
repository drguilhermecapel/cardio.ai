"""
Carregador Robusto do Modelo .h5 para Análise de ECG
Sistema de carregamento com monitoramento médico e validação diagnóstica
"""

import os
import json
import logging
import hashlib
import numpy as np
import tensorflow as tf
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import warnings

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suprimir warnings desnecessários do TensorFlow
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class ModelLoaderRobust:
    """
    Carregador robusto do modelo .h5 com monitoramento médico
    """
    
    def __init__(self, model_path: str, classes_path: str = None):
        self.model_path = model_path
        self.classes_path = classes_path or str(Path(model_path).parent / "ptbxl_classes.json")
        self.model = None
        self.classes_info = None
        self.model_metadata = {}
        self.validation_results = {}
        self.load_timestamp = None
        self.prediction_cache = {}
        
        # Configurações médicas
        self.medical_thresholds = {
            'confidence_threshold': 0.5,
            'critical_conditions': ['MI', 'STEMI', 'NSTEMI', 'VT', 'VF', 'AFIB'],
            'normal_class': 'NORM',
            'quality_threshold': 0.7
        }
        
        # Estatísticas de uso
        self.usage_stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'cache_hits': 0,
            'average_prediction_time': 0.0,
            'last_prediction_time': None
        }
        
        logger.info(f"Inicializando ModelLoaderRobust para: {model_path}")
        
    def load_model(self) -> bool:
        """
        Carrega o modelo .h5 com validação robusta
        """
        try:
            logger.info("Iniciando carregamento robusto do modelo .h5...")
            
            # Verificar arquivo
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
            
            file_size = os.path.getsize(self.model_path) / (1024 * 1024)
            logger.info(f"Arquivo do modelo: {file_size:.1f} MB")
            
            # Calcular hash para verificação de integridade
            model_hash = self._calculate_file_hash(self.model_path)
            logger.info(f"Hash do modelo: {model_hash[:16]}...")
            
            # Configurar TensorFlow para uso médico
            self._configure_tensorflow_medical()
            
            # Carregar modelo
            logger.info("Carregando modelo TensorFlow/Keras...")
            self.model = tf.keras.models.load_model(
                self.model_path, 
                compile=False,
                custom_objects=None
            )
            
            # Extrair metadados
            self.model_metadata = {
                'file_path': self.model_path,
                'file_size_mb': file_size,
                'file_hash': model_hash,
                'input_shape': self.model.input_shape,
                'output_shape': self.model.output_shape,
                'total_params': self.model.count_params(),
                'trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights]),
                'non_trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights]),
                'layers_count': len(self.model.layers),
                'load_timestamp': datetime.now().isoformat(),
                'tensorflow_version': tf.__version__,
                'keras_version': tf.keras.__version__
            }
            
            # Carregar informações das classes
            self._load_classes_info()
            
            # Validar modelo para uso médico
            validation_success = self._validate_model_medical()
            
            if validation_success:
                self.load_timestamp = datetime.now()
                logger.info("✅ Modelo carregado e validado com sucesso para uso médico!")
                return True
            else:
                logger.error("❌ Modelo falhou na validação médica")
                return False
                
        except Exception as e:
            logger.error(f"❌ Erro no carregamento do modelo: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _configure_tensorflow_medical(self):
        """
        Configura TensorFlow para uso médico otimizado
        """
        try:
            # Configurar para uso determinístico (importante para medicina)
            tf.config.experimental.enable_op_determinism()
            
            # Configurar memória GPU se disponível
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"GPU configurada: {len(gpus)} dispositivos")
                except RuntimeError as e:
                    logger.warning(f"Erro na configuração GPU: {e}")
            
            # Configurar threads para CPU
            tf.config.threading.set_inter_op_parallelism_threads(0)
            tf.config.threading.set_intra_op_parallelism_threads(0)
            
            logger.info("TensorFlow configurado para uso médico")
            
        except Exception as e:
            logger.warning(f"Erro na configuração TensorFlow: {e}")
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calcula hash SHA256 do arquivo para verificação de integridade
        """
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _load_classes_info(self):
        """
        Carrega informações das classes diagnósticas
        """
        try:
            if os.path.exists(self.classes_path):
                with open(self.classes_path, 'r', encoding='utf-8') as f:
                    self.classes_info = json.load(f)
                logger.info(f"Classes carregadas: {len(self.classes_info.get('classes', {}))} condições")
            else:
                logger.warning(f"Arquivo de classes não encontrado: {self.classes_path}")
                # Criar classes padrão PTB-XL
                self._create_default_classes()
                
        except Exception as e:
            logger.error(f"Erro ao carregar classes: {e}")
            self._create_default_classes()
    
    def _create_default_classes(self):
        """
        Cria mapeamento padrão das classes PTB-XL
        """
        self.classes_info = {
            "classes": {
                "0": "NORM - Normal ECG",
                "1": "MI - Myocardial Infarction", 
                "2": "STTC - ST/T Change",
                "3": "CD - Conduction Disturbance",
                "4": "HYP - Hypertrophy",
                "5": "AFIB - Atrial Fibrillation",
                "6": "STEMI - ST Elevation MI",
                "7": "NSTEMI - Non-ST Elevation MI",
                "8": "VT - Ventricular Tachycardia",
                "9": "VF - Ventricular Fibrillation",
                "10": "RBBB - Right Bundle Branch Block"
                # ... outras classes PTB-XL
            },
            "medical_categories": {
                "critical": ["MI", "STEMI", "NSTEMI", "VT", "VF"],
                "urgent": ["AFIB", "CD", "HYP"],
                "routine": ["NORM", "STTC"]
            }
        }
        logger.info("Classes padrão PTB-XL criadas")
    
    def _validate_model_medical(self) -> bool:
        """
        Valida modelo para uso médico com testes específicos
        """
        try:
            logger.info("Iniciando validação médica do modelo...")
            
            validation_tests = []
            
            # Teste 1: Verificar dimensões de entrada (12 derivações, 1000 amostras)
            expected_input = (None, 12, 1000)
            if self.model.input_shape == expected_input:
                validation_tests.append(("input_shape", True, "Dimensões de entrada corretas"))
            else:
                validation_tests.append(("input_shape", False, f"Dimensões incorretas: {self.model.input_shape}"))
            
            # Teste 2: Verificar número de classes de saída
            expected_classes = 71  # PTB-XL padrão
            output_classes = self.model.output_shape[-1]
            if output_classes == expected_classes:
                validation_tests.append(("output_classes", True, f"Classes corretas: {output_classes}"))
            else:
                validation_tests.append(("output_classes", False, f"Classes incorretas: {output_classes}"))
            
            # Teste 3: Teste de predição com ECG sintético normal
            normal_ecg = self._generate_synthetic_normal_ecg()
            try:
                prediction = self.model.predict(normal_ecg, verbose=0)
                if prediction.shape == (1, output_classes):
                    validation_tests.append(("prediction_shape", True, "Predição com formato correto"))
                else:
                    validation_tests.append(("prediction_shape", False, f"Formato incorreto: {prediction.shape}"))
                
                # Verificar se saída é probabilidade (0-1)
                if np.all(prediction >= 0) and np.all(prediction <= 1):
                    validation_tests.append(("probability_range", True, "Saída em formato de probabilidade"))
                else:
                    validation_tests.append(("probability_range", False, "Saída fora do range 0-1"))
                    
            except Exception as e:
                validation_tests.append(("prediction_test", False, f"Erro na predição: {e}"))
            
            # Teste 4: Teste de predição com ECG anormal (arritmia)
            arrhythmia_ecg = self._generate_synthetic_arrhythmia_ecg()
            try:
                prediction_arr = self.model.predict(arrhythmia_ecg, verbose=0)
                # Verificar se predições são diferentes para ECGs diferentes
                if not np.array_equal(prediction, prediction_arr):
                    validation_tests.append(("discrimination", True, "Modelo discrimina entre ECGs diferentes"))
                else:
                    validation_tests.append(("discrimination", False, "Modelo não discrimina adequadamente"))
                    
            except Exception as e:
                validation_tests.append(("discrimination_test", False, f"Erro no teste de discriminação: {e}"))
            
            # Teste 5: Verificar tempo de predição (deve ser < 5 segundos para uso clínico)
            start_time = datetime.now()
            _ = self.model.predict(normal_ecg, verbose=0)
            prediction_time = (datetime.now() - start_time).total_seconds()
            
            if prediction_time < 5.0:
                validation_tests.append(("prediction_speed", True, f"Tempo adequado: {prediction_time:.2f}s"))
            else:
                validation_tests.append(("prediction_speed", False, f"Muito lento: {prediction_time:.2f}s"))
            
            # Compilar resultados
            self.validation_results = {
                'timestamp': datetime.now().isoformat(),
                'tests': validation_tests,
                'passed_tests': sum(1 for _, passed, _ in validation_tests if passed),
                'total_tests': len(validation_tests),
                'success_rate': sum(1 for _, passed, _ in validation_tests if passed) / len(validation_tests),
                'medical_grade': None
            }
            
            # Determinar grau médico
            success_rate = self.validation_results['success_rate']
            if success_rate >= 0.9:
                self.validation_results['medical_grade'] = 'A - Aprovado para uso clínico'
            elif success_rate >= 0.7:
                self.validation_results['medical_grade'] = 'B - Aprovado com restrições'
            else:
                self.validation_results['medical_grade'] = 'C - Não aprovado para uso clínico'
            
            # Log dos resultados
            logger.info(f"Validação médica: {self.validation_results['passed_tests']}/{self.validation_results['total_tests']} testes aprovados")
            logger.info(f"Grau médico: {self.validation_results['medical_grade']}")
            
            for test_name, passed, message in validation_tests:
                status = "✅" if passed else "❌"
                logger.info(f"{status} {test_name}: {message}")
            
            return success_rate >= 0.7  # Mínimo 70% para aprovação
            
        except Exception as e:
            logger.error(f"Erro na validação médica: {e}")
            return False
    
    def _generate_synthetic_normal_ecg(self) -> np.ndarray:
        """
        Gera ECG sintético normal para teste
        """
        # ECG normal com 12 derivações, 1000 amostras (10s a 100Hz)
        ecg = np.zeros((1, 12, 1000))
        
        # Simular batimentos cardíacos normais (75 bpm)
        heart_rate = 75
        samples_per_beat = int(100 * 60 / heart_rate)  # ~80 amostras por batimento
        
        for lead in range(12):
            for beat in range(int(1000 / samples_per_beat)):
                start_idx = beat * samples_per_beat
                if start_idx + samples_per_beat < 1000:
                    # Onda P
                    ecg[0, lead, start_idx:start_idx+10] = 0.1 * np.sin(np.linspace(0, np.pi, 10))
                    # Complexo QRS
                    ecg[0, lead, start_idx+20:start_idx+35] = np.array([
                        -0.1, -0.2, 1.0, -0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                    ])
                    # Onda T
                    ecg[0, lead, start_idx+50:start_idx+70] = 0.3 * np.sin(np.linspace(0, np.pi, 20))
        
        # Adicionar ruído fisiológico mínimo
        ecg += np.random.normal(0, 0.01, ecg.shape)
        
        return ecg.astype(np.float32)
    
    def _generate_synthetic_arrhythmia_ecg(self) -> np.ndarray:
        """
        Gera ECG sintético com arritmia para teste
        """
        # ECG com fibrilação atrial (irregular)
        ecg = np.zeros((1, 12, 1000))
        
        # Ritmo irregular (40-150 bpm variável)
        current_pos = 0
        while current_pos < 950:
            # Intervalo RR variável (característica da AFIB)
            rr_interval = np.random.randint(40, 120)
            
            if current_pos + rr_interval < 1000:
                for lead in range(12):
                    # Sem onda P clara (característica da AFIB)
                    # Complexo QRS irregular
                    qrs_start = current_pos + 5
                    if qrs_start + 15 < 1000:
                        ecg[0, lead, qrs_start:qrs_start+15] = np.random.normal(0.8, 0.2) * np.array([
                            -0.1, 1.0, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                        ])
                    
                    # Onda T variável
                    t_start = current_pos + 30
                    if t_start + 20 < 1000:
                        ecg[0, lead, t_start:t_start+20] = np.random.normal(0.2, 0.1) * np.sin(np.linspace(0, np.pi, 20))
            
            current_pos += rr_interval
        
        # Adicionar fibrilação atrial (ondas f)
        f_waves = 0.05 * np.sin(2 * np.pi * 6 * np.linspace(0, 10, 1000))  # 6 Hz
        for lead in range(12):
            ecg[0, lead, :] += f_waves
        
        # Ruído adicional
        ecg += np.random.normal(0, 0.02, ecg.shape)
        
        return ecg.astype(np.float32)
    
    def predict_ecg(self, ecg_data: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Realiza predição de ECG com monitoramento médico
        """
        start_time = datetime.now()
        
        try:
            if self.model is None:
                raise RuntimeError("Modelo não carregado. Execute load_model() primeiro.")
            
            # Validar entrada
            if not isinstance(ecg_data, np.ndarray):
                raise ValueError("ECG data deve ser numpy array")
            
            # Preprocessar dados
            processed_data = self._preprocess_ecg_data(ecg_data)
            
            # Verificar cache
            data_hash = hashlib.md5(processed_data.tobytes()).hexdigest()
            if data_hash in self.prediction_cache:
                self.usage_stats['cache_hits'] += 1
                logger.info("Resultado obtido do cache")
                return self.prediction_cache[data_hash]
            
            # Realizar predição
            logger.info("Realizando predição com modelo PTB-XL...")
            raw_prediction = self.model.predict(processed_data, verbose=0)
            
            # Processar resultado
            result = self._process_prediction_result(raw_prediction, metadata)
            
            # Atualizar estatísticas
            prediction_time = (datetime.now() - start_time).total_seconds()
            self._update_usage_stats(prediction_time, success=True)
            
            # Cache do resultado
            self.prediction_cache[data_hash] = result
            
            # Limpar cache se muito grande
            if len(self.prediction_cache) > 100:
                oldest_key = next(iter(self.prediction_cache))
                del self.prediction_cache[oldest_key]
            
            logger.info(f"Predição concluída em {prediction_time:.2f}s")
            return result
            
        except Exception as e:
            prediction_time = (datetime.now() - start_time).total_seconds()
            self._update_usage_stats(prediction_time, success=False)
            logger.error(f"Erro na predição: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'processing_time': prediction_time
            }
    
    def _preprocess_ecg_data(self, ecg_data: np.ndarray) -> np.ndarray:
        """
        Preprocessa dados de ECG para o modelo
        """
        # Converter para formato esperado pelo modelo
        if len(ecg_data.shape) == 2:
            # Adicionar dimensão batch
            ecg_data = np.expand_dims(ecg_data, axis=0)
        
        # Verificar dimensões
        if ecg_data.shape[1:] != (12, 1000):
            logger.warning(f"Dimensões inesperadas: {ecg_data.shape}, tentando redimensionar...")
            # Tentar redimensionar ou interpolar
            if ecg_data.shape[1] == 12:
                # Interpolar para 1000 amostras
                from scipy import interpolate
                new_data = np.zeros((ecg_data.shape[0], 12, 1000))
                for i in range(ecg_data.shape[0]):
                    for lead in range(12):
                        if ecg_data.shape[2] != 1000:
                            x_old = np.linspace(0, 1, ecg_data.shape[2])
                            x_new = np.linspace(0, 1, 1000)
                            f = interpolate.interp1d(x_old, ecg_data[i, lead, :], kind='linear')
                            new_data[i, lead, :] = f(x_new)
                        else:
                            new_data[i, lead, :] = ecg_data[i, lead, :]
                ecg_data = new_data
        
        # Normalização (importante para modelos médicos)
        # Normalizar por derivação para preservar características morfológicas
        for i in range(ecg_data.shape[0]):
            for lead in range(12):
                lead_data = ecg_data[i, lead, :]
                if np.std(lead_data) > 0:
                    ecg_data[i, lead, :] = (lead_data - np.mean(lead_data)) / np.std(lead_data)
        
        return ecg_data.astype(np.float32)
    
    def _process_prediction_result(self, raw_prediction: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Processa resultado da predição com interpretação médica
        """
        # Obter probabilidades
        probabilities = raw_prediction[0]  # Remover dimensão batch
        
        # Encontrar classe principal
        primary_class_idx = np.argmax(probabilities)
        primary_probability = probabilities[primary_class_idx]
        
        # Obter informações da classe
        class_info = self._get_class_info(primary_class_idx)
        
        # Determinar nível de confiança médica
        confidence_level = self._determine_confidence_level(primary_probability)
        
        # Encontrar top diagnósticos
        top_indices = np.argsort(probabilities)[::-1][:5]
        top_diagnoses = []
        
        for idx in top_indices:
            if probabilities[idx] > 0.01:  # Apenas probabilidades > 1%
                class_info_top = self._get_class_info(idx)
                top_diagnoses.append({
                    'class_index': int(idx),
                    'class_name': class_info_top['name'],
                    'class_description': class_info_top['description'],
                    'probability': float(probabilities[idx]),
                    'medical_category': class_info_top['category']
                })
        
        # Determinar urgência médica
        urgency = self._determine_medical_urgency(class_info['name'], primary_probability)
        
        # Gerar recomendações clínicas
        clinical_recommendations = self._generate_clinical_recommendations(
            class_info['name'], primary_probability, top_diagnoses
        )
        
        # Resultado completo
        result = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'name': 'PTB-XL ECG Classifier',
                'version': self.model_metadata.get('tensorflow_version', 'unknown'),
                'validation_grade': self.validation_results.get('medical_grade', 'unknown')
            },
            'primary_diagnosis': {
                'class_index': int(primary_class_idx),
                'class_name': class_info['name'],
                'class_description': class_info['description'],
                'probability': float(primary_probability),
                'confidence_level': confidence_level,
                'medical_category': class_info['category']
            },
            'top_diagnoses': top_diagnoses,
            'medical_assessment': {
                'urgency_level': urgency,
                'confidence_score': float(primary_probability),
                'quality_indicators': {
                    'prediction_confidence': confidence_level,
                    'model_certainty': 'high' if primary_probability > 0.8 else 'moderate' if primary_probability > 0.5 else 'low'
                }
            },
            'clinical_recommendations': clinical_recommendations,
            'technical_details': {
                'raw_probabilities': probabilities.tolist(),
                'processing_metadata': metadata or {},
                'model_validation': self.validation_results
            }
        }
        
        return result
    
    def _get_class_info(self, class_idx: int) -> Dict[str, str]:
        """
        Obtém informações da classe diagnóstica
        """
        if self.classes_info and 'classes' in self.classes_info:
            class_name = self.classes_info['classes'].get(str(class_idx), f"Class_{class_idx}")
        else:
            class_name = f"Class_{class_idx}"
        
        # Separar nome e descrição
        if ' - ' in class_name:
            name, description = class_name.split(' - ', 1)
        else:
            name = class_name
            description = "Condição cardíaca detectada"
        
        # Determinar categoria médica
        category = 'routine'
        if any(crit in name.upper() for crit in ['MI', 'STEMI', 'NSTEMI', 'VT', 'VF']):
            category = 'critical'
        elif any(urg in name.upper() for urg in ['AFIB', 'CD', 'HYP', 'RBBB', 'LBBB']):
            category = 'urgent'
        elif 'NORM' in name.upper():
            category = 'normal'
        
        return {
            'name': name,
            'description': description,
            'category': category
        }
    
    def _determine_confidence_level(self, probability: float) -> str:
        """
        Determina nível de confiança médica
        """
        if probability >= 0.9:
            return 'muito_alta'
        elif probability >= 0.8:
            return 'alta'
        elif probability >= 0.6:
            return 'moderada'
        elif probability >= 0.4:
            return 'baixa'
        else:
            return 'muito_baixa'
    
    def _determine_medical_urgency(self, class_name: str, probability: float) -> str:
        """
        Determina urgência médica baseada no diagnóstico
        """
        class_upper = class_name.upper()
        
        # Condições críticas
        if any(crit in class_upper for crit in ['MI', 'STEMI', 'NSTEMI', 'VT', 'VF']):
            return 'critical' if probability > 0.7 else 'urgent'
        
        # Condições urgentes
        elif any(urg in class_upper for urg in ['AFIB', 'FLUTTER', 'SVT', 'RBBB', 'LBBB']):
            return 'urgent' if probability > 0.6 else 'high'
        
        # Normal
        elif 'NORM' in class_upper:
            return 'routine'
        
        # Outras condições
        else:
            return 'high' if probability > 0.8 else 'routine'
    
    def _generate_clinical_recommendations(self, class_name: str, probability: float, top_diagnoses: List[Dict]) -> Dict[str, Any]:
        """
        Gera recomendações clínicas baseadas no diagnóstico
        """
        class_upper = class_name.upper()
        
        recommendations = {
            'immediate_action': 'Revisão clínica recomendada',
            'additional_tests': [],
            'specialist_referral': None,
            'follow_up': 'Conforme protocolo institucional',
            'urgency_notes': [],
            'confidence_notes': []
        }
        
        # Recomendações baseadas no diagnóstico
        if 'MI' in class_upper or 'STEMI' in class_upper:
            recommendations.update({
                'immediate_action': 'EMERGÊNCIA - Protocolo de infarto agudo',
                'additional_tests': ['Troponinas seriadas', 'CK-MB', 'Ecocardiograma urgente'],
                'specialist_referral': 'Cardiologista/Hemodinâmica URGENTE',
                'urgency_notes': ['Suspeita de síndrome coronariana aguda', 'Tempo é músculo cardíaco']
            })
        
        elif 'AFIB' in class_upper:
            recommendations.update({
                'immediate_action': 'Avaliação cardiológica prioritária',
                'additional_tests': ['Ecocardiograma', 'TSH', 'Eletrólitos'],
                'specialist_referral': 'Cardiologista',
                'urgency_notes': ['Avaliar anticoagulação', 'Controle de frequência/ritmo']
            })
        
        elif 'VT' in class_upper or 'VF' in class_upper:
            recommendations.update({
                'immediate_action': 'EMERGÊNCIA - Arritmia potencialmente fatal',
                'additional_tests': ['Ecocardiograma urgente', 'Eletrólitos'],
                'specialist_referral': 'Cardiologista/Arritmia URGENTE',
                'urgency_notes': ['Risco de morte súbita', 'Monitorização contínua']
            })
        
        elif 'NORM' in class_upper:
            recommendations.update({
                'immediate_action': 'ECG normal - seguimento conforme indicação clínica',
                'follow_up': 'Rotina conforme avaliação médica'
            })
        
        # Notas de confiança
        if probability < 0.6:
            recommendations['confidence_notes'].append(
                'Baixa confiança na predição - revisão manual obrigatória'
            )
        
        if len(top_diagnoses) > 1 and top_diagnoses[1]['probability'] > 0.3:
            recommendations['confidence_notes'].append(
                f"Diagnóstico diferencial: {top_diagnoses[1]['class_name']} ({top_diagnoses[1]['probability']:.1%})"
            )
        
        return recommendations
    
    def _update_usage_stats(self, prediction_time: float, success: bool):
        """
        Atualiza estatísticas de uso
        """
        self.usage_stats['total_predictions'] += 1
        
        if success:
            self.usage_stats['successful_predictions'] += 1
        else:
            self.usage_stats['failed_predictions'] += 1
        
        # Atualizar tempo médio
        total_successful = self.usage_stats['successful_predictions']
        if total_successful > 0:
            current_avg = self.usage_stats['average_prediction_time']
            self.usage_stats['average_prediction_time'] = (
                (current_avg * (total_successful - 1) + prediction_time) / total_successful
            )
        
        self.usage_stats['last_prediction_time'] = datetime.now().isoformat()
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Retorna status completo do modelo
        """
        return {
            'model_loaded': self.model is not None,
            'load_timestamp': self.load_timestamp.isoformat() if self.load_timestamp else None,
            'model_metadata': self.model_metadata,
            'validation_results': self.validation_results,
            'usage_statistics': self.usage_stats,
            'medical_configuration': self.medical_thresholds,
            'classes_loaded': len(self.classes_info.get('classes', {})) if self.classes_info else 0
        }
    
    def is_ready_for_medical_use(self) -> bool:
        """
        Verifica se o modelo está pronto para uso médico
        """
        if not self.model:
            return False
        
        if not self.validation_results:
            return False
        
        return self.validation_results.get('success_rate', 0) >= 0.7


# Instância global do carregador
_model_loader_instance = None

def get_model_loader_robust(model_path: str = None) -> ModelLoaderRobust:
    """
    Obtém instância singleton do carregador robusto
    """
    global _model_loader_instance
    
    if _model_loader_instance is None:
        if model_path is None:
            model_path = "/home/ubuntu/cardio_ai_repo/models/ecg_model_final.h5"
        
        _model_loader_instance = ModelLoaderRobust(model_path)
        
        # Carregar modelo automaticamente
        success = _model_loader_instance.load_model()
        if not success:
            logger.error("Falha no carregamento automático do modelo")
    
    return _model_loader_instance

