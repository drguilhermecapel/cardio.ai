import torch
import wfdb
import numpy as np
import pandas as pd
import os
import zipfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
import ast

from .base_dataset import BaseECGDataset
from ..config.dataset_configs import get_dataset_config

logger = logging.getLogger(__name__)


class PTBXLDataset(BaseECGDataset):
    """Dataset para o PTB-XL ECG Database"""
    
    def __init__(self, data_path: Union[str, Path], sample_limit: Optional[int] = None, **kwargs):
        self.dataset_config = get_dataset_config("ptbxl")
        self.sampling_rate = kwargs.get("sampling_rate", self.dataset_config.sampling_rate)
        self.sample_limit = sample_limit
        self.data_path = Path(data_path)
        
        # Baixar dados se necessário (mantido para compatibilidade com o script de download)
        # self.download_data() # Comentado para evitar download automático ao instanciar dataset
        
        # Chamar construtor da classe pai
        super().__init__(data_path=self.data_path, **kwargs)
        
    def download_data(self):
        """Baixa os dados reais do dataset PTB-XL se não existirem. (Chamado apenas pelo script de download) """
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Verificar se os dados já existem
        metadata_path = self.data_path / "ptbxl_database.csv"
        records_path = self.data_path / f"records{self.sampling_rate}"
        
        if metadata_path.exists() and records_path.exists():
            logger.info("Dados do PTB-XL já existem, pulando download.")
            return
            
        logger.info("Baixando dataset PTB-XL...")
        
        # URL do arquivo ZIP do PTB-XL
        zip_url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
        zip_path = self.data_path / "ptb-xl.zip"
        
        try:
            # Download do arquivo ZIP
            if not zip_path.exists():
                logger.info(f"Baixando de {zip_url}...")
                urllib.request.urlretrieve(
                    zip_url, 
                    zip_path,
                    reporthook=self._download_progress
                )
                logger.info("Download concluído!")
            
            # Extrair arquivo ZIP
            logger.info("Extraindo arquivos...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_path)
                
            # Mover arquivos da pasta extraída para a pasta principal
            extracted_folder = self.data_path / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
            if extracted_folder.exists():
                for item in extracted_folder.iterdir():
                    target = self.data_path / item.name
                    if not target.exists():
                        item.rename(target)
                extracted_folder.rmdir()
                
            # Remover arquivo ZIP para economizar espaço
            zip_path.unlink()
            logger.info("Dataset PTB-XL baixado e extraído com sucesso!")
            
        except Exception as e:
            logger.error(f"Erro ao baixar PTB-XL: {e}")
            logger.info("Criando arquivos dummy para permitir execução...")
            self._create_dummy_files()
            
    def _download_progress(self, block_num, block_size, total_size):
        """Callback para mostrar progresso do download"""
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100)
        logger.info(f"Download: {percent:.1f}% ({downloaded/1024/1024:.1f}MB/{total_size/1024/1024:.1f}MB)")
        
    def _create_dummy_files(self):
        """Cria arquivos dummy para testes"""
        # Criar ptbxl_database.csv dummy
        ptbxl_db_path = self.data_path / "ptbxl_database.csv"
        if not ptbxl_db_path.exists():
            dummy_data = pd.DataFrame({
                'ecg_id': [1, 2, 3, 4, 5],
                'filename_hr': [f'records500/00000/0000{i}' for i in range(1, 6)],
                'filename_lr': [f'records100/00000/0000{i}' for i in range(1, 6)],
                'scp_codes': ['{"NORM": 100}', '{"MI": 100}', '{"STTC": 100}', '{"CD": 100}', '{"HYP": 100}'],
                'diagnostic_superclass': ['NORM', 'MI', 'STTC', 'CD', 'HYP'],
                'age': [60, 50, 45, 70, 55],
                'sex': [0, 1, 0, 1, 0],
                'report': ['Normal ECG', 'Myocardial Infarction', 'ST/T Change', 'Conduction Disturbance', 'Hypertrophy']
            })
            dummy_data.to_csv(ptbxl_db_path, index=False)
            logger.info(f"Arquivo dummy {ptbxl_db_path} criado.")

        # Criar scp_statements.csv dummy
        scp_statements_path = self.data_path / "scp_statements.csv"
        if not scp_statements_path.exists():
            scp_data = pd.DataFrame({
                'diagnostic_class': ['NORM', 'MI', 'STTC', 'CD', 'HYP'],
                'diagnostic': [1, 1, 1, 1, 1],
                'description': ['Normal ECG', 'Myocardial infarction', 'ST/T change', 'Conduction disturbance', 'Hypertrophy']
            })
            scp_data.to_csv(scp_statements_path, index=False)
            logger.info(f"Arquivo dummy {scp_statements_path} criado.")

        # Criar diretório records500 e alguns arquivos WFDB dummy
        records_path = self.data_path / f"records{self.sampling_rate}"
        records_path.mkdir(parents=True, exist_ok=True)
        
        # Criar subdiretório
        subdir = records_path / "00000"
        subdir.mkdir(exist_ok=True)
        
        # Criar alguns registros WFDB dummy
        for i in range(1, 6):
            record_name = f"0000{i}"
            # Criar sinal dummy (10 segundos, 12 derivações)
            signal = np.random.randn(self.sampling_rate * 10, 12) * 0.1
            
            # Salvar usando wfdb
            wfdb.wrsamp(
                record_name=record_name,
                fs=self.sampling_rate,
                units=["mV"]*12,
                sig_name=["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
                p_signal=signal,
                fmt=["16"]*12,
                write_dir=str(subdir)
            )
            
        logger.info(f"Registros dummy criados em {records_path}")

    def load_metadata(self):
        """Carrega metadados do dataset PTB-XL"""
        metadata_path = self.data_path / "ptbxl_database.csv"
        if not metadata_path.exists():
            # Se o arquivo de metadados não existe, tenta criar dummies
            logger.warning(f"Arquivo de metadados não encontrado em {metadata_path}. Tentando criar arquivos dummy.")
            self._create_dummy_files()
            # Após criar dummies, verifica novamente
            if not metadata_path.exists():
                raise FileNotFoundError(f"Arquivo de metadados não encontrado em {metadata_path} e falha ao criar dummies.")
            
        logger.info("Carregando metadados do PTB-XL...")
        self.raw_metadata = pd.read_csv(metadata_path, index_col='ecg_id')
        
        # Carregar statements para mapear códigos SCP
        scp_path = self.data_path / "scp_statements.csv"
        if scp_path.exists():
            self.scp_statements = pd.read_csv(scp_path, index_col=0)
        else:
            logger.warning("scp_statements.csv não encontrado")
            self.scp_statements = None
            
        # Processar labels
        self.process_labels()
        
        # Criar splits
        self.create_splits()
        
    def process_labels(self):
        """Processa labels do PTB-XL"""
        # Converter string de códigos SCP para dicionário
        self.raw_metadata['scp_codes_dict'] = self.raw_metadata['scp_codes'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else {}
        )
        
        # Usar superclasses se disponível
        if 'diagnostic_superclass' in self.raw_metadata.columns:
            self.raw_metadata['primary_label'] = self.raw_metadata['diagnostic_superclass']
        else:
            # Extrair label principal dos códigos SCP
            self.raw_metadata['primary_label'] = self.raw_metadata['scp_codes_dict'].apply(
                lambda x: max(x.items(), key=lambda item: item[1])[0] if x else 'NORM'
            )
            
        # Mapear para índices numéricos
        self.label_map = {label: idx for idx, label in enumerate(self.dataset_config.classes)}
        self.raw_metadata['label_idx'] = self.raw_metadata['primary_label'].map(
            lambda x: self.label_map.get(x, len(self.label_map))
        )
        
    def create_splits(self):
        """Cria splits de treino/validação/teste"""
        # PTB-XL tem splits recomendados
        # Por simplicidade, vamos dividir aleatoriamente
        n_samples = len(self.raw_metadata)
        
        # Aplicar limite de amostras se especificado
        if self.sample_limit and n_samples > self.sample_limit:
            self.raw_metadata = self.raw_metadata.sample(n=self.sample_limit, random_state=42)
            n_samples = self.sample_limit
            logger.info(f"Limitando dataset a {self.sample_limit} amostras")
        
        # Criar índices aleatórios
        indices = np.random.permutation(n_samples)
        
        # Dividir em 70/15/15
        train_end = int(0.7 * n_samples)
        val_end = int(0.85 * n_samples)
        
        self.train_indices = indices[:train_end]
        self.val_indices = indices[train_end:val_end]
        self.test_indices = indices[val_end:]
        
        # Definir samples baseado no split
        if self.split == "train":
            self.samples = self.raw_metadata.iloc[self.train_indices]
        elif self.split == "val":
            self.samples = self.raw_metadata.iloc[self.val_indices]
        else:  # test
            self.samples = self.raw_metadata.iloc[self.test_indices]
            
        self.labels = self.samples['label_idx'].values
        logger.info(f"Carregadas {len(self.samples)} amostras para split {self.split}")
        
    def load_signal(self, idx: int) -> Tuple[np.ndarray, Dict]:
        """Carrega sinal ECG do arquivo WFDB e metadados"""
        row = self.samples.iloc[idx]
        
        # Determinar caminho do arquivo
        if self.sampling_rate == 100:
            filename = row['filename_lr']
        else:
            filename = row['filename_hr']
            
        filepath = self.data_path / filename
        
        metadata = {
            'age': row.get('age', -1),
            'sex': row.get('sex', -1),
            'ecg_id': row.name, # ecg_id é o índice do dataframe
            'sampling_rate': self.sampling_rate # Adicionar sampling_rate aos metadados
        }

        try:
            # Carregar registro WFDB
            record = wfdb.rdrecord(str(filepath))
            signal = record.p_signal
            
            # O pré-processamento de padding/truncating e transposição
            # será feito na classe base (BaseECGDataset.preprocess_signal)
            
            return signal.astype(np.float32), metadata
            
        except Exception as e:
            logger.warning(f"Erro ao carregar {filepath}: {e}. Retornando sinal dummy.")
            # Retornar sinal dummy e metadados
            dummy_signal = np.random.randn(self.target_length, self.num_leads).astype(np.float32) * 0.1
            return dummy_signal, metadata
            
    def _get_label_map(self) -> Dict[str, int]:
        """Retorna mapeamento de labels para índices"""
        return self.label_map




