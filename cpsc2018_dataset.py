
"""
Implementação do dataset China Physiological Signal Challenge 2018 (CPSC2018)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import scipy.io

from .base_dataset import BaseECGDataset
from ..config.dataset_configs import get_dataset_config

logger = logging.getLogger(__name__)


class CPSC2018Dataset(BaseECGDataset):
    """Dataset para o China Physiological Signal Challenge 2018"""
    
    def __init__(self, data_path: Union[str, Path], **kwargs):
        super().__init__(data_path, **kwargs)
        self.dataset_config = get_dataset_config("cpsc2018")
        self._check_data_path()
        
    def load_metadata(self):
        """Carrega metadados do dataset CPSC2018"""
        # O CPSC2018 possui um arquivo CSV com metadados (REFERENCE.csv)
        # e os sinais em formato .mat (MATLAB)
        
        metadata_path = self.data_path / "REFERENCE.csv"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Arquivo de metadados REFERENCE.csv não encontrado em {self.data_path}")
            
        self.metadata = pd.read_csv(metadata_path, header=None, names=["record_name", "label_code"])
        
        # Mapeamento de códigos de label para nomes de classes
        # O dataset CPSC2018 usa códigos numéricos para as classes
        # Precisamos de um mapeamento para os nomes das classes
        # Assumindo que o mapeamento é fornecido ou inferido:
        # Exemplo: 0->Normal, 1->AF, etc. (ajustar conforme o REFERENCE.csv e a documentação)
        
        # Para este exemplo, vamos usar um mapeamento simplificado
        # Você precisará ajustar isso com base na documentação real do CPSC2018
        self.label_code_to_name = {
            0: "Normal",
            1: "AF",
            2: "I-AVB",
            3: "LBBB",
            4: "RBBB",
            5: "PAC",
            6: "PVC",
            7: "STD",
            8: "STE",
            # Adicione mais mapeamentos conforme necessário
        }
        
        self.metadata["label_name"] = self.metadata["label_code"].map(self.label_code_to_name)
        
        # Mapear nomes de classes para inteiros
        self.class_to_idx = {cls: i for i, cls in enumerate(self.dataset_config.classes)}
        self.metadata["label"] = self.metadata["label_name"].map(self.class_to_idx)
        
        # Remover amostras com labels desconhecidas ou nulas
        self.metadata.dropna(subset=["label"], inplace=True)
        self.metadata["label"] = self.metadata["label"].astype(int)
        
        self.samples = self.metadata["record_name"].tolist()
        self.labels = self.metadata["label"].tolist()
        
        logger.info(f"CPSC2018 metadata loaded. Found {len(self.samples)} ECGs.")

    def load_signal(self, idx: int) -> Tuple[np.ndarray, Dict]:
        """Carrega sinal ECG do registro especificado"""
        record_name = self.samples[idx]
        mat_file_path = self.data_path / f"{record_name}.mat"
        
        try:
            mat_data = scipy.io.loadmat(mat_file_path)
            # O sinal ECG geralmente está em uma chave como 'val' ou 'ECG'
            # Verifique a estrutura do arquivo .mat do CPSC2018
            signal = mat_data["val"].astype(np.float32) # Ajuste a chave conforme necessário
            
            # CPSC2018 tem 12 derivações
            if signal.shape[0] != self.num_leads:
                # Se o dataset tem menos derivações, preenche com zeros
                if signal.shape[0] < self.num_leads:
                    padding = np.zeros((self.num_leads - signal.shape[0], signal.shape[1]))
                    signal = np.vstack((signal, padding))
                # Se o dataset tem mais derivações, seleciona as primeiras
                else:
                    signal = signal[:self.num_leads, :]
            
            metadata = {
                "record_name": record_name,
                "sampling_rate": self.dataset_config.sampling_rate,
                "num_leads": self.num_leads,
                "signal_length": signal.shape[1],
                "label_code": self.metadata.loc[self.metadata["record_name"] == record_name, "label_code"].iloc[0],
                "label_name": self.metadata.loc[self.metadata["record_name"] == record_name, "label_name"].iloc[0]
            }
            
            return signal, metadata
        
        except Exception as e:
            logger.error(f"Erro ao carregar sinal para {record_name}: {e}")
            # Retorna um sinal de zeros e metadados vazios em caso de erro
            return np.zeros((self.num_leads, self.target_length)), {"error": str(e)}


