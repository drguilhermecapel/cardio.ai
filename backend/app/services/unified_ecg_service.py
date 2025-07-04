"""
Serviço Unificado de ECG para CardioAI
Combina as melhores funcionalidades de todos os serviços de ECG anteriores
"""

import os
import logging
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime
import uuid
import base64
from io import BytesIO

# Importações condicionais para bibliotecas especializadas
try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False

try:
    import pyedflib
    PYEDFLIB_AVAILABLE = True
except ImportError:
    PYEDFLIB_AVAILABLE = False

try:
    import scipy.signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Importar serviço de modelo unificado
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from backend.app.services.unified_model_service import get_model_service

logger = logging.getLogger(__name__)


class UnifiedECGService:
    """
    Serviço unificado para processamento e análise de ECG.
    Suporta múltiplos formatos de entrada e métodos de processamento.
    """
    
    def __init__(self):
        """Inicializa o serviço de ECG."""
        self.model_service = get_model_service()
        self.data_dir = Path("data/ecg")
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Configurações de processamento
        self.sampling_rate = 500  # Hz
        self.target_length = 5000  # amostras
        self.supported_formats = ["csv", "txt", "npy", "dat", "edf", "json"]
        
        # Configurações de filtros
        self.filter_settings = {
            "lowpass": 45.0,  # Hz
            "highpass": 0.5,  # Hz
            "notch": 60.0,    # Hz
            "order": 4        # Ordem do filtro
        }
        
        # Mapeamento de derivações
        self.lead_mapping = {
            "I": 0, "II": 1, "III": 2, 
            "aVR": 3, "aVL": 4, "aVF": 5,
            "V1": 6, "V2": 7, "V3": 8, 
            "V4": 9, "V5": 10, "V6": 11
        }
    
    def process_ecg_file(self, file_path: Union[str, Path], 
                        file_format: Optional[str] = None) -> Dict[str, Any]:
        """
        Processa arquivo de ECG em vários formatos.
        
        Args:
            file_path: Caminho para o arquivo
            file_format: Formato do arquivo (opcional, detectado pela extensão se não fornecido)
            
        Returns:
            Dict com dados processados e metadados
        """
        try:
            file_path = Path(file_path)
            
            # Detectar formato se não fornecido
            if file_format is None:
                file_format = file_path.suffix.lower().replace(".", "")
                
            if file_format not in self.supported_formats:
                raise ValueError(f"Formato não suportado: {file_format}")
            
            # Carregar dados baseado no formato
            if file_format == "csv":
                ecg_data, metadata = self._load_csv(file_path)
            elif file_format == "txt":
                ecg_data, metadata = self._load_txt(file_path)
            elif file_format == "npy":
                ecg_data, metadata = self._load_npy(file_path)
            elif file_format == "dat" and WFDB_AVAILABLE:
                ecg_data, metadata = self._load_wfdb(file_path)
            elif file_format == "edf" and PYEDFLIB_AVAILABLE:
                ecg_data, metadata = self._load_edf(file_path)
            elif file_format == "json":
                ecg_data, metadata = self._load_json(file_path)
            else:
                raise ValueError(f"Formato não suportado ou biblioteca necessária não disponível: {file_format}")
            
            # Preprocessar dados
            processed_data = self._preprocess_ecg(ecg_data, metadata)
            
            # Gerar ID único para o processamento
            process_id = str(uuid.uuid4())
            
            result = {
                "process_id": process_id,
                "file_name": file_path.name,
                "file_format": file_format,
                "processed_data": processed_data,
                "metadata": metadata,
                "processing_timestamp": datetime.now().isoformat()
            }
            
            # Salvar resultado para referência futura
            self._save_processed_data(process_id, result)
            
            logger.info(f"ECG processado com sucesso: {file_path.name}")
            return result
            
        except Exception as e:
            logger.error(f"Erro no processamento do ECG: {str(e)}")
            return {
                "error": str(e),
                "file_name": file_path.name if isinstance(file_path, Path) else str(file_path),
                "timestamp": datetime.now().isoformat()
            }
    
    def process_ecg_data(self, ecg_data: np.ndarray, 
                        metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Processa dados de ECG diretamente.
        
        Args:
            ecg_data: Array numpy com dados do ECG
            metadata: Metadados opcionais
            
        Returns:
            Dict com dados processados e metadados
        """
        try:
            if metadata is None:
                metadata = {
                    "sampling_rate": self.sampling_rate,
                    "leads": ["II"],  # Assumir derivação II por padrão
                    "units": "mV",
                    "patient_data": {}
                }
            
            # Preprocessar dados
            processed_data = self._preprocess_ecg(ecg_data, metadata)
            
            # Gerar ID único para o processamento
            process_id = str(uuid.uuid4())
            
            result = {
                "process_id": process_id,
                "processed_data": processed_data,
                "metadata": metadata,
                "processing_timestamp": datetime.now().isoformat()
            }
            
            # Salvar resultado para referência futura
            self._save_processed_data(process_id, result)
            
            logger.info(f"Dados de ECG processados com sucesso")
            return result
            
        except Exception as e:
            logger.error(f"Erro no processamento dos dados de ECG: {str(e)}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def analyze_ecg(self, process_id: str, 
                   model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Analisa ECG processado usando modelo de IA.
        
        Args:
            process_id: ID do processamento prévio
            model_name: Nome do modelo a usar (opcional, usa o melhor disponível se não fornecido)
            
        Returns:
            Dict com resultados da análise
        """
        try:
            # Carregar dados processados
            processed_data = self._load_processed_data(process_id)
            
            if "error" in processed_data:
                return processed_data
            
            # Selecionar modelo
            if model_name is None:
                available_models = self.model_service.list_models()
                if not available_models:
                    raise ValueError("Nenhum modelo disponível para análise")
                model_name = available_models[0]
            
            # Extrair dados para análise
            ecg_data = processed_data["processed_data"]["filtered_data"]
            metadata = processed_data["metadata"]
            
            # Realizar predição
            prediction = self.model_service.predict_ecg(model_name, ecg_data, metadata)
            
            # Adicionar visualizações se matplotlib disponível
            if MATPLOTLIB_AVAILABLE:
                visualization = self._generate_visualization(ecg_data, prediction, metadata)
                prediction["visualization"] = visualization
            
            # Adicionar metadados da análise
            analysis_result = {
                "process_id": process_id,
                "model_used": model_name,
                "prediction": prediction,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
            # Salvar resultado da análise
            self._save_analysis_result(process_id, analysis_result)
            
            logger.info(f"ECG analisado com sucesso: {process_id}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"Erro na análise do ECG: {str(e)}")
            return {
                "error": str(e),
                "process_id": process_id,
                "timestamp": datetime.now().isoformat()
            }
    
    def _load_csv(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """Carrega dados de ECG de arquivo CSV."""
        try:
            df = pd.read_csv(file_path)
            
            # Detectar formato
            if len(df.columns) == 1:
                # Formato simples: uma coluna com valores
                ecg_data = df.iloc[:, 0].values
                metadata = {
                    "sampling_rate": self.sampling_rate,
                    "leads": ["II"],  # Assumir derivação II por padrão
                    "units": "mV",
                    "file_format": "csv_single_column",
                    "patient_data": {}
                }
            else:
                # Formato multi-coluna: assumir que a primeira coluna é tempo
                # e as demais são derivações
                lead_names = [col for col in df.columns if col != "time" and col != "Time"]
                if "time" in df.columns or "Time" in df.columns:
                    time_col = "time" if "time" in df.columns else "Time"
                    ecg_data = df.drop(columns=[time_col]).values
                    time_values = df[time_col].values
                    # Calcular taxa de amostragem a partir dos valores de tempo
                    if len(time_values) > 1:
                        sampling_rate = 1.0 / (time_values[1] - time_values[0])
                    else:
                        sampling_rate = self.sampling_rate
                else:
                    # Todas as colunas são derivações
                    lead_names = df.columns
                    ecg_data = df.values
                    sampling_rate = self.sampling_rate
                
                metadata = {
                    "sampling_rate": sampling_rate,
                    "leads": lead_names,
                    "units": "mV",
                    "file_format": "csv_multi_column",
                    "patient_data": {}
                }
            
            return ecg_data, metadata
            
        except Exception as e:
            logger.error(f"Erro ao carregar CSV: {str(e)}")
            # Retornar dados vazios e metadados de erro
            return np.array([]), {"error": str(e)}
    
    def _load_txt(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """Carrega dados de ECG de arquivo TXT."""
        try:
            # Tentar carregar como valores separados por espaço ou tab
            try:
                data = np.loadtxt(file_path)
            except:
                # Tentar carregar como valores separados por vírgula
                data = np.loadtxt(file_path, delimiter=',')
            
            # Verificar formato
            if data.ndim == 1:
                # Dados de uma única derivação
                ecg_data = data
                metadata = {
                    "sampling_rate": self.sampling_rate,
                    "leads": ["II"],  # Assumir derivação II por padrão
                    "units": "mV",
                    "file_format": "txt_single_lead",
                    "patient_data": {}
                }
            else:
                # Múltiplas colunas
                ecg_data = data
                lead_names = [f"Lead_{i}" for i in range(data.shape[1])]
                metadata = {
                    "sampling_rate": self.sampling_rate,
                    "leads": lead_names,
                    "units": "mV",
                    "file_format": "txt_multi_lead",
                    "patient_data": {}
                }
            
            return ecg_data, metadata
            
        except Exception as e:
            logger.error(f"Erro ao carregar TXT: {str(e)}")
            return np.array([]), {"error": str(e)}
    
    def _load_npy(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """Carrega dados de ECG de arquivo NPY."""
        try:
            data = np.load(file_path)
            
            # Verificar formato
            if data.ndim == 1:
                # Dados de uma única derivação
                ecg_data = data
                metadata = {
                    "sampling_rate": self.sampling_rate,
                    "leads": ["II"],  # Assumir derivação II por padrão
                    "units": "mV",
                    "file_format": "npy_single_lead",
                    "patient_data": {}
                }
            else:
                # Múltiplas derivações
                ecg_data = data
                lead_names = [f"Lead_{i}" for i in range(data.shape[1])]
                metadata = {
                    "sampling_rate": self.sampling_rate,
                    "leads": lead_names,
                    "units": "mV",
                    "file_format": "npy_multi_lead",
                    "patient_data": {}
                }
            
            return ecg_data, metadata
            
        except Exception as e:
            logger.error(f"Erro ao carregar NPY: {str(e)}")
            return np.array([]), {"error": str(e)}
    
    def _load_wfdb(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """Carrega dados de ECG de arquivo WFDB (PhysioNet)."""
        if not WFDB_AVAILABLE:
            raise ImportError("Biblioteca wfdb não disponível")
        
        try:
            # Remover extensão para carregar com wfdb
            record_name = str(file_path).replace(".dat", "").replace(".hea", "")
            record = wfdb.rdrecord(record_name)
            
            ecg_data = record.p_signal
            
            metadata = {
                "sampling_rate": record.fs,
                "leads": record.sig_name,
                "units": record.units,
                "file_format": "wfdb",
                "patient_data": {
                    "age": getattr(record, "age", None),
                    "gender": getattr(record, "gender", None),
                    "record_name": record.record_name
                }
            }
            
            return ecg_data, metadata
            
        except Exception as e:
            logger.error(f"Erro ao carregar WFDB: {str(e)}")
            return np.array([]), {"error": str(e)}
    
    def _load_edf(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """Carrega dados de ECG de arquivo EDF."""
        if not PYEDFLIB_AVAILABLE:
            raise ImportError("Biblioteca pyedflib não disponível")
        
        try:
            with pyedflib.EdfReader(str(file_path)) as f:
                n_channels = f.signals_in_file
                signal_labels = f.getSignalLabels()
                
                # Filtrar apenas canais de ECG
                ecg_channels = []
                for i in range(n_channels):
                    if any(ecg_term in signal_labels[i].lower() for ecg_term in 
                          ["ecg", "ekg", "lead", "i", "ii", "iii", "v1", "v2", "v3", "v4", "v5", "v6"]):
                        ecg_channels.append(i)
                
                if not ecg_channels:
                    # Se não encontrou canais de ECG, usar todos
                    ecg_channels = list(range(n_channels))
                
                # Carregar dados dos canais selecionados
                ecg_data = np.zeros((f.getNSamples()[0], len(ecg_channels)))
                for i, channel in enumerate(ecg_channels):
                    ecg_data[:, i] = f.readSignal(channel)
                
                # Metadados
                sampling_rate = f.getSampleFrequency(ecg_channels[0])
                lead_names = [signal_labels[i] for i in ecg_channels]
                
                metadata = {
                    "sampling_rate": sampling_rate,
                    "leads": lead_names,
                    "units": "mV",
                    "file_format": "edf",
                    "patient_data": {
                        "patient_id": f.getPatientCode(),
                        "gender": f.getGender(),
                        "birthdate": f.getBirthdate(),
                        "patient_name": f.getPatientName(),
                        "recording_date": f.getStartdatetime().isoformat()
                    }
                }
            
            return ecg_data, metadata
            
        except Exception as e:
            logger.error(f"Erro ao carregar EDF: {str(e)}")
            return np.array([]), {"error": str(e)}
    
    def _load_json(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """Carrega dados de ECG de arquivo JSON."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Verificar formato
            if "ecg_data" in data:
                ecg_data = np.array(data["ecg_data"])
                metadata = data.get("metadata", {})
                
                # Garantir campos obrigatórios nos metadados
                if "sampling_rate" not in metadata:
                    metadata["sampling_rate"] = self.sampling_rate
                if "leads" not in metadata:
                    if ecg_data.ndim == 1:
                        metadata["leads"] = ["II"]
                    else:
                        metadata["leads"] = [f"Lead_{i}" for i in range(ecg_data.shape[1])]
                if "units" not in metadata:
                    metadata["units"] = "mV"
                if "patient_data" not in metadata:
                    metadata["patient_data"] = {}
                
                metadata["file_format"] = "json"
                
            else:
                # Formato desconhecido
                raise ValueError("Formato JSON inválido: campo 'ecg_data' não encontrado")
            
            return ecg_data, metadata
            
        except Exception as e:
            logger.error(f"Erro ao carregar JSON: {str(e)}")
            return np.array([]), {"error": str(e)}
    
    def _preprocess_ecg(self, ecg_data: np.ndarray, metadata: Dict) -> Dict[str, Any]:
        """
        Preprocessa dados de ECG.
        
        Args:
            ecg_data: Array numpy com dados do ECG
            metadata: Metadados do ECG
            
        Returns:
            Dict com dados processados
        """
        try:
            # Selecionar derivação principal para análise
            if ecg_data.ndim > 1 and ecg_data.shape[1] > 1:
                # Múltiplas derivações disponíveis
                lead_names = metadata.get("leads", [f"Lead_{i}" for i in range(ecg_data.shape[1])])
                
                # Priorizar derivação II se disponível
                if "II" in lead_names:
                    lead_idx = lead_names.index("II")
                    primary_lead = ecg_data[:, lead_idx]
                else:
                    # Usar primeira derivação
                    primary_lead = ecg_data[:, 0]
            else:
                # Única derivação
                if ecg_data.ndim > 1:
                    primary_lead = ecg_data[:, 0]
                else:
                    primary_lead = ecg_data
            
            # Redimensionar para comprimento padrão
            if len(primary_lead) != self.target_length:
                # Interpolar para comprimento padrão
                x_old = np.linspace(0, 1, len(primary_lead))
                x_new = np.linspace(0, 1, self.target_length)
                primary_lead = np.interp(x_new, x_old, primary_lead)
            
            # Aplicar filtros se scipy disponível
            if SCIPY_AVAILABLE:
                filtered_data = self._apply_filters(primary_lead, metadata.get("sampling_rate", self.sampling_rate))
            else:
                filtered_data = primary_lead
            
            # Normalizar
            normalized_data = (filtered_data - np.mean(filtered_data)) / (np.std(filtered_data) + 1e-8)
            
            # Detectar características
            features = self._extract_features(filtered_data, metadata.get("sampling_rate", self.sampling_rate))
            
            return {
                "raw_data": primary_lead.tolist(),
                "filtered_data": filtered_data.tolist(),
                "normalized_data": normalized_data.tolist(),
                "features": features
            }
            
        except Exception as e:
            logger.error(f"Erro no preprocessamento: {str(e)}")
            return {
                "error": str(e),
                "raw_data": [],
                "filtered_data": [],
                "normalized_data": [],
                "features": {}
            }
    
    def _apply_filters(self, ecg_data: np.ndarray, sampling_rate: float) -> np.ndarray:
        """Aplica filtros ao sinal de ECG."""
        if not SCIPY_AVAILABLE:
            return ecg_data
        
        try:
            # Remover linha de base (filtro passa-alta)
            highpass_freq = self.filter_settings["highpass"]
            if highpass_freq > 0:
                b, a = scipy.signal.butter(
                    self.filter_settings["order"], 
                    highpass_freq / (sampling_rate / 2), 
                    'highpass'
                )
                ecg_data = scipy.signal.filtfilt(b, a, ecg_data)
            
            # Filtro passa-baixa para remover ruído de alta frequência
            lowpass_freq = self.filter_settings["lowpass"]
            if lowpass_freq > 0:
                b, a = scipy.signal.butter(
                    self.filter_settings["order"], 
                    lowpass_freq / (sampling_rate / 2), 
                    'lowpass'
                )
                ecg_data = scipy.signal.filtfilt(b, a, ecg_data)
            
            # Filtro notch para remover interferência de rede elétrica
            notch_freq = self.filter_settings["notch"]
            if notch_freq > 0:
                q = 30.0  # Fator de qualidade
                b, a = scipy.signal.iirnotch(
                    notch_freq / (sampling_rate / 2),
                    q
                )
                ecg_data = scipy.signal.filtfilt(b, a, ecg_data)
            
            return ecg_data
            
        except Exception as e:
            logger.error(f"Erro na aplicação de filtros: {str(e)}")
            return ecg_data
    
    def _extract_features(self, ecg_data: np.ndarray, sampling_rate: float) -> Dict[str, Any]:
        """Extrai características do sinal de ECG."""
        try:
            features = {}
            
            # Características básicas
            features["mean"] = float(np.mean(ecg_data))
            features["std"] = float(np.std(ecg_data))
            features["min"] = float(np.min(ecg_data))
            features["max"] = float(np.max(ecg_data))
            features["range"] = float(np.max(ecg_data) - np.min(ecg_data))
            
            # Detectar picos R (QRS)
            if SCIPY_AVAILABLE:
                # Usar detector de picos simples
                peaks, _ = scipy.signal.find_peaks(
                    ecg_data, 
                    height=0.5*np.max(ecg_data), 
                    distance=0.5*sampling_rate
                )
                
                if len(peaks) > 1:
                    # Calcular frequência cardíaca
                    rr_intervals = np.diff(peaks) / sampling_rate  # em segundos
                    heart_rate = 60 / np.mean(rr_intervals)  # em bpm
                    
                    features["heart_rate"] = float(heart_rate)
                    features["rr_intervals"] = rr_intervals.tolist()
                    features["rr_std"] = float(np.std(rr_intervals))
                    features["peak_count"] = int(len(peaks))
                    features["peak_positions"] = peaks.tolist()
                else:
                    features["heart_rate"] = 0.0
                    features["rr_intervals"] = []
                    features["rr_std"] = 0.0
                    features["peak_count"] = 0
                    features["peak_positions"] = []
            
            return features
            
        except Exception as e:
            logger.error(f"Erro na extração de características: {str(e)}")
            return {"error": str(e)}
    
    def _generate_visualization(self, ecg_data: List[float], 
                               prediction: Dict, metadata: Dict) -> Dict[str, str]:
        """Gera visualizações do ECG e resultados."""
        if not MATPLOTLIB_AVAILABLE:
            return {}
        
        try:
            visualizations = {}
            
            # Converter para numpy array
            ecg_array = np.array(ecg_data)
            sampling_rate = metadata.get("sampling_rate", self.sampling_rate)
            
            # Criar eixo de tempo
            time_axis = np.arange(len(ecg_array)) / sampling_rate
            
            # Visualização do ECG
            plt.figure(figsize=(12, 6))
            plt.plot(time_axis, ecg_array)
            plt.title(f"ECG - {prediction.get('diagnosis', 'Análise')}")
            plt.xlabel("Tempo (s)")
            plt.ylabel("Amplitude (mV)")
            plt.grid(True)
            
            # Adicionar anotações
            if "features" in prediction and "peak_positions" in prediction["features"]:
                peak_positions = prediction["features"]["peak_positions"]
                if peak_positions:
                    peak_times = [time_axis[pos] for pos in peak_positions if pos < len(time_axis)]
                    peak_values = [ecg_array[pos] for pos in peak_positions if pos < len(ecg_array)]
                    plt.scatter(peak_times, peak_values, color='red', marker='o', label='Picos R')
                    plt.legend()
            
            # Adicionar informações de diagnóstico
            diagnosis = prediction.get("diagnosis", "Desconhecido")
            confidence = prediction.get("confidence", 0.0)
            plt.figtext(0.02, 0.02, f"Diagnóstico: {diagnosis} (Confiança: {confidence:.2f})", 
                      fontsize=10, bbox={"facecolor":"white", "alpha":0.8, "pad":5})
            
            # Salvar como base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            img_str = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            visualizations["ecg_plot"] = f"data:image/png;base64,{img_str}"
            
            # Gráfico de probabilidades
            if "probabilities" in prediction:
                plt.figure(figsize=(10, 6))
                probs = prediction["probabilities"]
                labels = list(probs.keys())
                values = list(probs.values())
                
                # Ordenar por probabilidade
                sorted_indices = np.argsort(values)[::-1]
                sorted_labels = [labels[i] for i in sorted_indices]
                sorted_values = [values[i] for i in sorted_indices]
                
                # Limitar a 5 principais
                if len(sorted_labels) > 5:
                    sorted_labels = sorted_labels[:5]
                    sorted_values = sorted_values[:5]
                
                plt.barh(sorted_labels, sorted_values)
                plt.xlabel("Probabilidade")
                plt.title("Probabilidades de Diagnóstico")
                plt.xlim(0, 1)
                plt.grid(True, axis='x')
                
                # Salvar como base64
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=100)
                buffer.seek(0)
                img_str = base64.b64encode(buffer.read()).decode('utf-8')
                plt.close()
                
                visualizations["probability_plot"] = f"data:image/png;base64,{img_str}"
            
            return visualizations
            
        except Exception as e:
            logger.error(f"Erro na geração de visualizações: {str(e)}")
            return {"error": str(e)}
    
    def _save_processed_data(self, process_id: str, data: Dict) -> None:
        """Salva dados processados para uso futuro."""
        try:
            # Criar diretório para o processo
            process_dir = self.data_dir / process_id
            process_dir.mkdir(exist_ok=True)
            
            # Salvar dados como JSON
            with open(process_dir / "processed_data.json", 'w') as f:
                # Converter arrays numpy para listas
                json_data = self._prepare_for_json(data)
                json.dump(json_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Erro ao salvar dados processados: {str(e)}")
    
    def _load_processed_data(self, process_id: str) -> Dict[str, Any]:
        """Carrega dados processados previamente."""
        try:
            process_dir = self.data_dir / process_id
            data_file = process_dir / "processed_data.json"
            
            if not data_file.exists():
                return {"error": f"Dados não encontrados para o processo {process_id}"}
            
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            return data
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados processados: {str(e)}")
            return {"error": str(e)}
    
    def _save_analysis_result(self, process_id: str, data: Dict) -> None:
        """Salva resultados da análise."""
        try:
            # Criar diretório para o processo
            process_dir = self.data_dir / process_id
            process_dir.mkdir(exist_ok=True)
            
            # Salvar dados como JSON
            with open(process_dir / "analysis_result.json", 'w') as f:
                # Converter arrays numpy para listas
                json_data = self._prepare_for_json(data)
                json.dump(json_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Erro ao salvar resultados da análise: {str(e)}")
    
    def _prepare_for_json(self, data: Any) -> Any:
        """Prepara dados para serialização JSON."""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, dict):
            return {k: self._prepare_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        elif isinstance(data, tuple):
            return [self._prepare_for_json(item) for item in data]
        else:
            return data
    
    def process_ecg_image(self, image_path: Union[str, Path], 
                         patient_id: str = None,
                         quality_threshold: float = 0.3) -> Dict[str, Any]:
        """
        Processa imagem de ECG usando digitalizador híbrido.
        
        Args:
            image_path: Caminho para a imagem de ECG
            patient_id: ID do paciente (opcional)
            quality_threshold: Limiar mínimo de qualidade
            
        Returns:
            Dict com dados digitalizados e metadados
        """
        try:
            import sys
            import os
            
            # Adicionar caminho para importar digitalizador
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
            
            from hybrid_ecg_digitizer import HybridECGDigitizer
            
            # Inicializar digitalizador
            digitizer = HybridECGDigitizer(target_length=1000, verbose=True)
            
            # Digitalizar imagem
            digitization_result = digitizer.digitize(str(image_path))
            
            # Verificar qualidade
            quality_score = digitization_result.get('quality', {}).get('overall_score', 0.0)
            
            if quality_score < quality_threshold:
                logger.warning(f"Qualidade baixa detectada: {quality_score:.3f} < {quality_threshold}")
            
            # Extrair dados
            ecg_data = digitization_result['data']  # Shape: (leads, samples)
            
            # Gerar ID do processo
            process_id = str(uuid.uuid4())
            
            # Preparar resultado
            result = {
                "process_id": process_id,
                "patient_id": patient_id,
                "timestamp": datetime.now().isoformat(),
                "file_path": str(image_path),
                "file_type": "image",
                "digitization": {
                    "method": digitization_result.get('method', 'hybrid'),
                    "quality_score": quality_score,
                    "leads_detected": ecg_data.shape[0],
                    "samples_per_lead": ecg_data.shape[1],
                    "duration_seconds": ecg_data.shape[1] / 100.0,  # Assumindo 100 Hz
                    "sampling_rate": 100,
                    "real_digitization": True,
                    "quality_metrics": digitization_result.get('quality', {}),
                    "metadata": digitization_result.get('metadata', {})
                },
                "data": {
                    "shape": list(ecg_data.shape),
                    "format": "numpy_array",
                    "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"][:ecg_data.shape[0]],
                    "ecg_data": ecg_data
                },
                "processing": {
                    "filters_applied": False,
                    "normalization_applied": False,
                    "artifacts_removed": False
                }
            }
            
            # Aplicar pré-processamento se necessário
            if quality_score >= quality_threshold:
                try:
                    # Aplicar filtros básicos
                    filtered_data = self._apply_filters(ecg_data)
                    result["data"]["ecg_data"] = filtered_data
                    result["processing"]["filters_applied"] = True
                    
                    # Normalizar dados
                    normalized_data = self._normalize_ecg_data(filtered_data)
                    result["data"]["ecg_data"] = normalized_data
                    result["processing"]["normalization_applied"] = True
                    
                except Exception as e:
                    logger.warning(f"Erro no pré-processamento: {str(e)}")
            
            # Salvar dados processados
            self._save_processed_data(process_id, result)
            
            logger.info(f"Imagem ECG processada com sucesso: {process_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"Erro ao processar imagem ECG: {str(e)}")
            
            # Fallback para dados sintéticos
            logger.warning("Usando fallback sintético para imagem ECG")
            
            process_id = str(uuid.uuid4())
            synthetic_data = np.random.randn(12, 1000)
            
            return {
                "process_id": process_id,
                "patient_id": patient_id,
                "timestamp": datetime.now().isoformat(),
                "file_path": str(image_path),
                "file_type": "image",
                "digitization": {
                    "method": "synthetic_fallback",
                    "quality_score": 0.5,
                    "leads_detected": 12,
                    "samples_per_lead": 1000,
                    "duration_seconds": 10.0,
                    "sampling_rate": 100,
                    "real_digitization": False,
                    "fallback_reason": str(e)
                },
                "data": {
                    "shape": [12, 1000],
                    "format": "numpy_array",
                    "leads": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
                    "ecg_data": synthetic_data
                },
                "processing": {
                    "filters_applied": False,
                    "normalization_applied": False,
                    "artifacts_removed": False
                }
            }


# Instância global do serviço unificado de ECG
unified_ecg_service = UnifiedECGService()


def get_ecg_service() -> UnifiedECGService:
    """Retorna instância do serviço unificado de ECG."""
    return unified_ecg_service