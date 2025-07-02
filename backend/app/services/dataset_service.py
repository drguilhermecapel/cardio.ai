"""
Serviço de integração com datasets públicos de ECG
Integrado com o sistema CardioAI Pro
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ..datasets import (
    ECGDatasetAnalyzer,
    ECGDatasetDownloader,
    ECGDatasetLoader,
    ECGRecord,
    prepare_ml_dataset,
)
from ..preprocessing import AdvancedECGPreprocessor


class DatasetService:
    """Serviço para gerenciar datasets públicos de ECG"""

    def __init__(self, base_dir: str = "ecg_datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

        self.downloader = ECGDatasetDownloader(str(self.base_dir))
        self.preprocessor = AdvancedECGPreprocessor()
        self.loader = ECGDatasetLoader(self.preprocessor)
        self.analyzer = ECGDatasetAnalyzer()

        self.logger = logging.getLogger(__name__)

    def download_dataset(self, dataset_name: str, **kwargs: Any) -> str | None:
        """
        Baixa um dataset específico

        Args:
            dataset_name: Nome do dataset ('mit-bih', 'ptb-xl', 'cpsc2018')
            **kwargs: Argumentos específicos para cada dataset

        Returns:
            Caminho do dataset baixado ou None se falhou
        """
        self.logger.info(f"Iniciando download do dataset: {dataset_name}")

        try:
            if dataset_name == "mit-bih":
                return self.downloader.download_mit_bih(**kwargs)
            elif dataset_name == "ptb-xl":
                return self.downloader.download_ptb_xl(**kwargs)
            elif dataset_name == "cpsc2018":
                return self.downloader.download_cpsc2018(**kwargs)
            else:
                self.logger.error(f"Dataset não suportado: {dataset_name}")
                return None

        except Exception as e:
            self.logger.error(f"Erro ao baixar {dataset_name}: {e}")
            return None

    def load_dataset(
        self,
        dataset_name: str,
        dataset_path: str,
        preprocess: bool = True,
        max_records: int | None = None,
    ) -> list[ECGRecord]:
        """
        Carrega um dataset com pré-processamento opcional

        Args:
            dataset_name: Nome do dataset
            dataset_path: Caminho para o dataset
            preprocess: Se True, aplica pré-processamento avançado
            max_records: Número máximo de registros

        Returns:
            Lista de ECGRecords
        """
        self.logger.info(f"Carregando dataset: {dataset_name}")

        try:
            if dataset_name == "mit-bih":
                return self.loader.load_mit_bih(
                    dataset_path, preprocess=preprocess, max_records=max_records
                )
            elif dataset_name == "ptb-xl":
                return self.loader.load_ptb_xl(
                    dataset_path, max_records=max_records, preprocess=preprocess
                )
            else:
                self.logger.error(f"Loader não implementado para: {dataset_name}")
                return []

        except Exception as e:
            self.logger.error(f"Erro ao carregar {dataset_name}: {e}")
            return []

    def analyze_dataset(
        self, records: list[ECGRecord], dataset_name: str
    ) -> dict[str, Any]:
        """
        Analisa estatísticas de um dataset

        Args:
            records: Lista de ECGRecords
            dataset_name: Nome do dataset

        Returns:
            Dicionário com estatísticas
        """
        return self.analyzer.analyze_dataset(records, dataset_name)

    def prepare_for_ml(
        self,
        records: list[ECGRecord],
        window_size: int = 3600,
        target_labels: list[str] | None = None,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
        """
        Prepara dataset para machine learning

        Args:
            records: Lista de ECGRecords
            window_size: Tamanho da janela em amostras
            target_labels: Labels específicos para filtrar

        Returns:
            X, y arrays para ML
        """
        return prepare_ml_dataset(records, window_size, target_labels)

    def quick_setup_mit_bih(
        self, num_records: int = 10
    ) -> tuple[list[ECGRecord] | None, dict[str, Any] | None]:
        """
        Setup rápido do MIT-BIH para testes

        Args:
            num_records: Número de registros para baixar

        Returns:
            Tupla (records, stats) ou (None, None) se falhou
        """
        try:
            record_names = [f"{i:03d}" for i in range(100, 100 + num_records)]

            dataset_path = self.download_dataset(
                "mit-bih", records_to_download=record_names
            )

            if not dataset_path:
                return None, None

            records = self.load_dataset("mit-bih", dataset_path, preprocess=True)

            if not records:
                return None, None

            stats = self.analyze_dataset(records, "MIT-BIH Quick Setup")

            self.logger.info(f"✓ Setup rápido concluído: {len(records)} registros")
            return records, stats

        except Exception as e:
            self.logger.error(f"Erro no setup rápido: {e}")
            return None, None

    def get_available_datasets(self) -> dict[str, dict[str, Any]]:
        """Retorna informações sobre datasets disponíveis"""
        return self.downloader.DATASETS_INFO

    def create_unified_dataset(
        self,
        datasets: dict[str, list[ECGRecord]],
        output_path: str = "unified_ecg_dataset.h5",
    ) -> str:
        """
        Cria dataset unificado em formato HDF5

        Args:
            datasets: Dicionário com datasets
            output_path: Caminho de saída

        Returns:
            Caminho do arquivo criado
        """
        return self.loader.create_unified_dataset(datasets, output_path)

    def validate_environment(self) -> dict[str, bool]:
        """
        Valida se o ambiente tem todas as dependências necessárias

        Returns:
            Dicionário com status das dependências
        """
        dependencies = {}

        try:
            import wfdb  # noqa: F401

            dependencies["wfdb"] = True
        except ImportError:
            dependencies["wfdb"] = False

        try:
            import h5py  # type: ignore # noqa: F401

            dependencies["h5py"] = True
        except ImportError:
            dependencies["h5py"] = False

        try:
            import pandas  # noqa: F401

            dependencies["pandas"] = True
        except ImportError:
            dependencies["pandas"] = False

        try:
            from ..preprocessing import AdvancedECGPreprocessor  # noqa: F401

            dependencies["advanced_preprocessor"] = True
        except ImportError:
            dependencies["advanced_preprocessor"] = False

        return dependencies
