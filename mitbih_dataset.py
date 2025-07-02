"""
Implementação do dataset MIT-BIH Arrhythmia Database
"""

import wfdb
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

from .base_dataset import BaseECGDataset
from ..config.dataset_configs import get_dataset_config

logger = logging.getLogger(__name__)


class MITBIHDataset(BaseECGDataset):
    """Dataset para o MIT-BIH Arrhythmia Database"""
    
    def __init__(self, data_path: Union[str, Path], **kwargs):
        super().__init__(data_path, **kwargs)
        self.dataset_config = get_dataset_config("mitbih")
        self._check_data_path()
        
    def load_metadata(self):
        """Carrega metadados do dataset MIT-BIH"""
        # MIT-BIH tem 48 registros. Os nomes dos registros são de 100 a 234.
        # Alguns registros são duplos (e.g., 201, 202) e outros são simples.
        # Para simplificar, vamos carregar todos os registros disponíveis no diretório.
        
        record_files = sorted(list(self.data_path.glob("*.dat")))
        if not record_files:
            raise FileNotFoundError(f"Nenhum arquivo .dat encontrado em {self.data_path}")
            
        self.records = [f.stem for f in record_files]
        
        # Para MIT-BIH, cada registro é uma 'amostra' longa. Vamos tratar cada registro como uma amostra.
        # As labels serão baseadas nas anotações de arritmia.
        
        self.samples = []
        self.labels = []
        
        # Mapeamento de anotações para classes numéricas
        # Simplificado para 5 classes principais (N, S, V, F, Q)
        # Ref: https://archive.physionet.org/physiobank/annotations.shtml
        self.annotation_map = {
            'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal
            'A': 1, 'a': 1, 'J': 1, 'S': 1,  # Supraventricular
            'V': 2, 'E': 2,  # Ventricular
            'F': 3,  # Fusion
            'Q': 4, '/': 4,  # Unknown / Paced
        }
        
        for record_name in self.records:
            record_path = self.data_path / record_name
            try:
                annotation = wfdb.rdann(str(record_path), extension='atr')
                # Coleta todas as anotações e mapeia para classes
                record_labels = [self.annotation_map.get(sym, 4) for sym in annotation.symbol]
                
                # Para este dataset, cada 'amostra' é um registro completo.
                # A label do registro pode ser a mais frequente ou uma representação agregada.
                # Para simplificar, vamos adicionar o registro como uma amostra e uma label dummy por enquanto.
                # O treinamento real pode focar em classificação de batimentos.
                self.samples.append(record_name)
                # Usar a classe mais frequente como label do registro (simplificado)
                if record_labels:
                    from collections import Counter
                    most_common = Counter(record_labels).most_common(1)
                    self.labels.append(most_common[0][0])
                else:
                    self.labels.append(4) # Unknown
                    
            except Exception as e:
                logger.warning(f"Não foi possível carregar anotações para {record_name}: {e}")
                continue
        
        logger.info(f"MIT-BIH metadata loaded. Found {len(self.samples)} records.")

    def load_signal(self, idx: int) -> Tuple[np.ndarray, Dict]:
        """Carrega sinal ECG do registro especificado"""
        record_name = self.samples[idx]
        npy_path = self.data_path / f"{record_name}.npy"
        
        try:
            if npy_path.exists():
                signal = self.load_npy_signal(npy_path)
            else:
                record_path = self.data_path / record_name
                record = wfdb.rdrecord(str(record_path))
                signal = record.p_signal.T  # Transpõe para (num_leads, num_samples)
            
            metadata = {
                "record_name": record_name,
                "sampling_rate": record.fs,
                "num_leads": record.n_sig,
                "signal_length": record.sig_len,
                "units": record.units,
                "lead_names": record.sig_name
            }
            
            # Assegura que o número de derivações corresponde ao esperado
            if signal.shape[0] != self.num_leads:
                # Se o dataset tem menos derivações, preenche com zeros
                if signal.shape[0] < self.num_leads:
                    padding = np.zeros((self.num_leads - signal.shape[0], signal.shape[1]))
                    signal = np.vstack((signal, padding))
                # Se o dataset tem mais derivações, seleciona as primeiras
                else:
                    signal = signal[:self.num_leads, :]
            
            return signal, metadata
        
        except Exception as e:
            logger.error(f"Erro ao carregar sinal para {record_name}: {e}")
            # Retorna um sinal de zeros e metadados vazios em caso de erro
            return np.zeros((self.num_leads, self.target_length)), {"error": str(e)}
