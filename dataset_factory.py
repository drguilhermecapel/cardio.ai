
"""
Fábrica para criação de instâncias de datasets de ECG
"""

from pathlib import Path
from typing import Union
from .base_dataset import BaseECGDataset
from .mitbih_dataset import MITBIHDataset
from .ptbxl_dataset import PTBXLDataset
from .cpsc2018_dataset import CPSC2018Dataset


class DatasetFactory:
    """Classe para criar instâncias de datasets de ECG"""
    
    @staticmethod
    def create_dataset(
        dataset_name: str,
        data_path: Union[str, Path],
        split: str = "train",
        **kwargs
    ) -> BaseECGDataset:
        """
        Cria e retorna uma instância de um dataset de ECG.
        
        Args:
            dataset_name: Nome do dataset (ex: "mitbih", "ptbxl", "cpsc2018")
            data_path: Caminho para o diretório raiz dos dados do dataset.
            split: O split do dataset a ser carregado ("train", "val", "test").
            **kwargs: Argumentos adicionais a serem passados para o construtor do dataset.
            
        Returns:
            Uma instância de BaseECGDataset ou uma de suas subclasses.
            
        Raises:
            ValueError: Se o dataset_name não for reconhecido.
        """
        
        dataset_name = dataset_name.lower()
        
        if dataset_name == "mitbih":
            return MITBIHDataset(data_path=data_path, split=split, **kwargs)
        elif dataset_name == "ptbxl":
            return PTBXLDataset(data_path=data_path, split=split, **kwargs)
        elif dataset_name == "cpsc2018":
            return CPSC2018Dataset(data_path=data_path, split=split, **kwargs)
        else:
            raise ValueError(f"Dataset {dataset_name} não suportado. "
                            f"Datasets disponíveis: mitbih, ptbxl, cpsc2018")


