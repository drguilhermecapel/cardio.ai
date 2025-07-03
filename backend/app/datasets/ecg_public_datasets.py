"""
Sistema de Integração com Datasets Públicos de ECG
Integrado com o pipeline de pré-processamento avançado do CardioAI Pro
"""

import json
import logging
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import h5py  # type: ignore
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import wfdb

    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False
    logging.warning("wfdb não disponível. Instale com: pip install wfdb")

try:

    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    logging.warning("PyWavelets não disponível. Instale com: pip install PyWavelets")

try:
    from ..preprocessing import AdvancedECGPreprocessor

    ADVANCED_PREPROCESSOR_AVAILABLE = True
except ImportError:
    ADVANCED_PREPROCESSOR_AVAILABLE = False
    logging.warning("AdvancedECGPreprocessor não disponível")


@dataclass
class ECGRecord:
    """Estrutura padronizada para registros de ECG"""

    signal: np.ndarray[np.float64, np.dtype[np.float64]]
    sampling_rate: int
    labels: list[str] = field(default_factory=list)
    patient_id: str = ""
    age: int | None = None
    sex: str | None = None
    leads: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    annotations: dict[str, Any] | None = None


class ECGDatasetDownloader:
    """Downloader para datasets públicos de ECG"""

    DATASETS_INFO = {
        "mit-bih": {
            "name": "MIT-BIH Arrhythmia Database",
            "url": "https://physionet.org/files/mitdb/1.0.0/",
            "description": "48 registros de ECG de 2 derivações com anotações de arritmias",
            "size": "~23MB",
            "records": 48,
            "sampling_rate": 360,
            "leads": 2,
            "duration": "30 min por registro",
        },
        "ptb-xl": {
            "name": "PTB-XL Database",
            "url": "https://physionet.org/files/ptb-xl/1.0.3/",
            "description": "21,799 ECGs de 12 derivações com diagnósticos clínicos",
            "size": "~3GB",
            "records": 21799,
            "sampling_rate": 500,
            "leads": 12,
            "duration": "10s por registro",
        },
        "cpsc2018": {
            "name": "CPSC 2018 Challenge",
            "url": "http://2018.icbeb.org/Challenge.html",
            "description": "6,877 ECGs de 12 derivações para detecção de arritmias",
            "size": "~1.5GB",
            "records": 6877,
            "sampling_rate": 500,
            "leads": 12,
            "duration": "Variável (6-60s)",
        },
        "mimic-iv-ecg": {
            "name": "MIMIC-IV-ECG Database",
            "url": "https://physionet.org/content/mimic-iv-ecg/1.0/",
            "description": "800,000+ ECGs de 12 derivações de pacientes de UTI",
            "size": "~50GB",
            "records": 800000,
            "sampling_rate": 500,
            "leads": 12,
            "duration": "10s por registro",
        },
        "icentia11k": {
            "name": "Icentia11k Database",
            "url": "https://physionet.org/content/icentia11k-continuous-ecg/1.0/",
            "description": "11,000 pacientes com ECG contínuo de 24h",
            "size": "~2TB",
            "records": 11000,
            "sampling_rate": 250,
            "leads": 1,
            "duration": "24h por paciente",
        },
        "physionet-challenge-2020": {
            "name": "PhysioNet Challenge 2020",
            "url": "https://physionet.org/content/challenge-2020/1.0.2/",
            "description": "43,101 ECGs de 12 derivações para classificação de arritmias",
            "size": "~6GB",
            "records": 43101,
            "sampling_rate": 500,
            "leads": 12,
            "duration": "Variável (6-300s)",
        },
        "physionet-challenge-2021": {
            "name": "PhysioNet Challenge 2021",
            "url": "https://physionet.org/content/challenge-2021/1.0.3/",
            "description": "88,253 ECGs de 12 derivações para diagnóstico de múltiplas condições",
            "size": "~12GB",
            "records": 88253,
            "sampling_rate": 500,
            "leads": 12,
            "duration": "Variável (6-300s)",
        },
    }

    def __init__(self, base_dir: str = "ecg_datasets"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def download_mit_bih(
        self,
        records_to_download: list[str] | None = None,
        force_redownload: bool = False,
    ) -> str | None:
        """
        Baixa o dataset MIT-BIH Arrhythmia

        Args:
            records_to_download: Lista de registros específicos (ex: ['100', '101'])
            force_redownload: Se True, redownload mesmo se já existir
        """
        if not WFDB_AVAILABLE:
            self.logger.error("wfdb não disponível. Instale com: pip install wfdb")
            return None

        dataset_dir = self.base_dir / "mit-bih"
        dataset_dir.mkdir(exist_ok=True)

        self.logger.info("Baixando MIT-BIH Arrhythmia Database...")

        if records_to_download is None:
            records_to_download = [
                "100",
                "101",
                "102",
                "103",
                "104",
                "105",
                "106",
                "107",
                "108",
                "109",
                "111",
                "112",
                "113",
                "114",
                "115",
                "116",
                "117",
                "118",
                "119",
                "121",
                "122",
                "123",
                "124",
                "200",
                "201",
                "202",
                "203",
                "205",
                "207",
                "208",
                "209",
                "210",
                "212",
                "213",
                "214",
                "215",
                "217",
                "219",
                "220",
                "221",
                "222",
                "223",
                "228",
                "230",
                "231",
                "232",
                "233",
                "234",
            ]

        try:
            for record_name in tqdm(records_to_download, desc="Baixando registros"):
                record_path = dataset_dir / record_name

                if record_path.with_suffix(".dat").exists() and not force_redownload:
                    continue

                wfdb.dl_database("mitdb", str(dataset_dir), records=[record_name])

            self.logger.info(f"✓ MIT-BIH baixado com sucesso em: {dataset_dir}")
            return str(dataset_dir)

        except Exception as e:
            self.logger.error(f"Erro ao baixar MIT-BIH: {e}")
            return None

    def download_ptb_xl(self, force_redownload: bool = False) -> str | None:
        """
        Baixa o dataset PTB-XL

        Args:
            force_redownload: Se True, redownload mesmo se já existir
        """
        dataset_dir = self.base_dir / "ptb-xl"
        dataset_dir.mkdir(exist_ok=True)

        expected_file = (
            dataset_dir
            / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
        )
        if expected_file.exists() and not force_redownload:
            self.logger.info(
                "PTB-XL já existe. Use force_redownload=True para redownload"
            )
            return str(dataset_dir)

        self.logger.info("Baixando PTB-XL Database (pode levar alguns minutos)...")

        url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"

        try:

            def download_hook(block_num: int, block_size: int, total_size: int) -> None:
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, downloaded * 100 / total_size)
                    print(
                        f"\rDownload: {percent:.1f}% ({downloaded/1024/1024:.1f}MB)",
                        end="",
                    )

            import urllib.request

            urllib.request.urlretrieve(url, expected_file, download_hook)
            print()  # Nova linha após download

            self.logger.info("Extraindo arquivos...")
            with zipfile.ZipFile(expected_file, "r") as zip_ref:
                zip_ref.extractall(dataset_dir)

            self.logger.info(f"✓ PTB-XL baixado e extraído em: {dataset_dir}")
            return str(dataset_dir)

        except Exception as e:
            self.logger.error(f"Erro ao baixar PTB-XL: {e}")
            return None

    def download_cpsc2018(self, force_redownload: bool = False) -> str | None:
        """
        Baixa o dataset CPSC 2018

        Args:
            force_redownload: Se True, redownload mesmo se já existir
        """
        dataset_dir = self.base_dir / "cpsc2018"
        dataset_dir.mkdir(exist_ok=True)

        self.logger.info("⚠️  CPSC 2018 requer download manual")
        self.logger.info("Visite: http://2018.icbeb.org/Challenge.html")
        self.logger.info("Baixe os arquivos e coloque em: " + str(dataset_dir))

        return str(dataset_dir)

    def download_mimic_iv_ecg(self, force_redownload: bool = False) -> str | None:
        """
        Baixa o dataset MIMIC-IV-ECG (requer credenciais PhysioNet)

        Args:
            force_redownload: Se True, redownload mesmo se já existir
        """
        dataset_dir = self.base_dir / "mimic-iv-ecg"
        dataset_dir.mkdir(exist_ok=True)

        self.logger.info("📋 MIMIC-IV-ECG requer credenciais PhysioNet")
        self.logger.info("1. Crie conta em: https://physionet.org/register/")
        self.logger.info("2. Complete o treinamento CITI")
        self.logger.info("3. Solicite acesso ao MIMIC-IV-ECG")
        self.logger.info("4. Use wget com suas credenciais:")
        self.logger.info(
            f"   wget -r -N -c -np --user YOUR_USERNAME --ask-password https://physionet.org/files/mimic-iv-ecg/1.0/ -P {dataset_dir}"
        )

        return str(dataset_dir)

    def download_icentia11k(self, force_redownload: bool = False) -> str | None:
        """
        Baixa o dataset Icentia11k (requer credenciais PhysioNet)

        Args:
            force_redownload: Se True, redownload mesmo se já existir
        """
        dataset_dir = self.base_dir / "icentia11k"
        dataset_dir.mkdir(exist_ok=True)

        self.logger.info("📋 Icentia11k requer credenciais PhysioNet")
        self.logger.info("1. Crie conta em: https://physionet.org/register/")
        self.logger.info("2. Complete o treinamento CITI")
        self.logger.info("3. Solicite acesso ao Icentia11k")
        self.logger.info("4. Use wget com suas credenciais:")
        self.logger.info(
            f"   wget -r -N -c -np --user YOUR_USERNAME --ask-password https://physionet.org/files/icentia11k-continuous-ecg/1.0/ -P {dataset_dir}"
        )

        return str(dataset_dir)

    def download_physionet_challenge_2020(
        self, force_redownload: bool = False
    ) -> str | None:
        """
        Baixa o dataset PhysioNet Challenge 2020

        Args:
            force_redownload: Se True, redownload mesmo se já existir
        """
        dataset_dir = self.base_dir / "physionet-challenge-2020"
        dataset_dir.mkdir(exist_ok=True)

        expected_file = dataset_dir / "challenge-2020-1.0.2.zip"
        if expected_file.exists() and not force_redownload:
            self.logger.info(
                "PhysioNet Challenge 2020 já existe. Use force_redownload=True para redownload"
            )
            return str(dataset_dir)

        self.logger.info("Baixando PhysioNet Challenge 2020...")

        url = "https://physionet.org/static/published-projects/challenge-2020/classification-of-12-lead-ecgs-the-physionet-computing-in-cardiology-challenge-2020-1.0.2.zip"

        try:

            def download_hook(block_num: int, block_size: int, total_size: int) -> None:
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, downloaded * 100 / total_size)
                    print(
                        f"\rDownload: {percent:.1f}% ({downloaded/1024/1024:.1f}MB)",
                        end="",
                    )

            import urllib.request

            urllib.request.urlretrieve(url, expected_file, download_hook)
            print()

            self.logger.info("Extraindo arquivos...")
            with zipfile.ZipFile(expected_file, "r") as zip_ref:
                zip_ref.extractall(dataset_dir)

            self.logger.info(
                f"✓ PhysioNet Challenge 2020 baixado e extraído em: {dataset_dir}"
            )
            return str(dataset_dir)

        except Exception as e:
            self.logger.error(f"Erro ao baixar PhysioNet Challenge 2020: {e}")
            return None

    def download_physionet_challenge_2021(
        self, force_redownload: bool = False
    ) -> str | None:
        """
        Baixa o dataset PhysioNet Challenge 2021

        Args:
            force_redownload: Se True, redownload mesmo se já existir
        """
        dataset_dir = self.base_dir / "physionet-challenge-2021"
        dataset_dir.mkdir(exist_ok=True)

        expected_file = dataset_dir / "challenge-2021-1.0.3.zip"
        if expected_file.exists() and not force_redownload:
            self.logger.info(
                "PhysioNet Challenge 2021 já existe. Use force_redownload=True para redownload"
            )
            return str(dataset_dir)

        self.logger.info("Baixando PhysioNet Challenge 2021...")

        url = "https://physionet.org/static/published-projects/challenge-2021/will-two-do-varying-dimensions-in-electrocardiography-the-physionet-computing-in-cardiology-challenge-2021-1.0.3.zip"

        try:

            def download_hook(block_num: int, block_size: int, total_size: int) -> None:
                downloaded = block_num * block_size
                if total_size > 0:
                    percent = min(100, downloaded * 100 / total_size)
                    print(
                        f"\rDownload: {percent:.1f}% ({downloaded/1024/1024:.1f}MB)",
                        end="",
                    )

            import urllib.request

            urllib.request.urlretrieve(url, expected_file, download_hook)
            print()

            self.logger.info("Extraindo arquivos...")
            with zipfile.ZipFile(expected_file, "r") as zip_ref:
                zip_ref.extractall(dataset_dir)

            self.logger.info(
                f"✓ PhysioNet Challenge 2021 baixado e extraído em: {dataset_dir}"
            )
            return str(dataset_dir)

        except Exception as e:
            self.logger.error(f"Erro ao baixar PhysioNet Challenge 2021: {e}")
            return None

    def download_all_datasets(
        self, force_redownload: bool = False
    ) -> dict[str, str | None]:
        """
        Baixa todos os datasets disponíveis

        Args:
            force_redownload: Se True, redownload mesmo se já existir

        Returns:
            Dicionário com nome do dataset -> caminho (ou None se falhou)
        """
        results = {}

        self.logger.info("🚀 Iniciando download de todos os datasets...")

        results["mit-bih"] = self.download_mit_bih(force_redownload=force_redownload)
        results["ptb-xl"] = self.download_ptb_xl(force_redownload=force_redownload)
        results["cpsc2018"] = self.download_cpsc2018(force_redownload=force_redownload)
        results["mimic-iv-ecg"] = self.download_mimic_iv_ecg(
            force_redownload=force_redownload
        )
        results["icentia11k"] = self.download_icentia11k(
            force_redownload=force_redownload
        )
        results["physionet-challenge-2020"] = self.download_physionet_challenge_2020(
            force_redownload=force_redownload
        )
        results["physionet-challenge-2021"] = self.download_physionet_challenge_2021(
            force_redownload=force_redownload
        )

        successful_downloads = [
            name for name, path in results.items() if path is not None
        ]
        failed_downloads = [name for name, path in results.items() if path is None]

        self.logger.info(
            f"✅ Downloads concluídos: {len(successful_downloads)}/{len(results)}"
        )
        if successful_downloads:
            self.logger.info(f"   Sucessos: {', '.join(successful_downloads)}")
        if failed_downloads:
            self.logger.warning(f"   Falhas: {', '.join(failed_downloads)}")

        return results

    def get_dataset_info(self, dataset_name: str) -> dict[str, Any]:
        return self.DATASETS_INFO.get(dataset_name, {})

    def list_available_datasets(self) -> None:
        """Lista todos os datasets disponíveis"""
        print("\n" + "=" * 60)
        print("DATASETS PÚBLICOS DE ECG DISPONÍVEIS")
        print("=" * 60)

        for name, info in self.DATASETS_INFO.items():
            print(f"\n📊 {info['name']} ({name})")
            print(f"   Descrição: {info['description']}")
            print(f"   Registros: {info['records']}")
            print(f"   Taxa de amostragem: {info['sampling_rate']} Hz")
            print(f"   Derivações: {info['leads']}")
            print(f"   Tamanho: {info['size']}")


class ECGDatasetLoader:
    """Carregador unificado para diferentes datasets"""

    def __init__(self, preprocessor: Optional["AdvancedECGPreprocessor"] = None):
        """
        Inicializa o carregador

        Args:
            preprocessor: Instância do AdvancedECGPreprocessor para pré-processamento
        """
        if preprocessor is None and ADVANCED_PREPROCESSOR_AVAILABLE:
            self.preprocessor: AdvancedECGPreprocessor | None = (
                AdvancedECGPreprocessor()
            )
        else:
            self.preprocessor = preprocessor

        self.label_mappings = self._initialize_label_mappings()
        self.logger = logging.getLogger(__name__)

    def _initialize_label_mappings(self) -> dict[str, Any]:
        """Inicializa mapeamentos de labels entre datasets"""
        return {
            "mit-bih": {
                "N": "normal",
                "L": "left_bundle_branch_block",
                "R": "right_bundle_branch_block",
                "A": "pac",  # Premature atrial contraction
                "a": "aberrated_pac",
                "J": "nodal_escape",
                "S": "pvc",  # Premature ventricular contraction
                "V": "pvc",
                "F": "fusion",
                "e": "atrial_escape",
                "j": "nodal_escape",
                "/": "paced",
                "Q": "unclassifiable",
            }
        }

    def load_mit_bih(
        self, dataset_path: str, preprocess: bool = True, max_records: int | None = None
    ) -> list[ECGRecord]:
        """
        Carrega registros do MIT-BIH

        Args:
            dataset_path: Caminho para o dataset
            preprocess: Se True, aplica pré-processamento avançado
            max_records: Número máximo de registros para carregar
        """
        if not WFDB_AVAILABLE:
            self.logger.error("wfdb não disponível")
            return []

        dataset_path_obj = Path(dataset_path)
        if not dataset_path_obj.exists():
            self.logger.error(f"Dataset não encontrado: {dataset_path}")
            return []

        header_files = list(dataset_path_obj.glob("*.hea"))
        if max_records:
            header_files = header_files[:max_records]

        ecg_records = []

        self.logger.info(f"Carregando {len(header_files)} registros do MIT-BIH...")

        for header_file in tqdm(header_files, desc="Carregando MIT-BIH"):
            try:
                record_name = header_file.stem

                record = wfdb.rdrecord(str(header_file.with_suffix("")))
                annotation = wfdb.rdann(str(header_file.with_suffix("")), "atr")

                signal_data = record.p_signal

                if preprocess and self.preprocessor:
                    try:
                        processed_signal, quality_metrics = (
                            self.preprocessor.advanced_preprocessing_pipeline(
                                signal_data[:, 0], clinical_mode=True
                            )
                        )

                        if quality_metrics["quality_score"] > 0.5:
                            signal_data = np.column_stack(
                                [processed_signal, signal_data[:, 1]]
                            )

                    except Exception as e:
                        self.logger.warning(
                            f"Erro no pré-processamento de {record_name}: {e}"
                        )

                labels = []
                unique_symbols = set(annotation.symbol)
                for symbol in unique_symbols:
                    if symbol in self.label_mappings["mit-bih"]:
                        labels.append(self.label_mappings["mit-bih"][symbol])
                    else:
                        labels.append(f"unknown_{symbol}")

                ecg_record = ECGRecord(
                    signal=signal_data,
                    sampling_rate=record.fs,
                    labels=labels,
                    patient_id=record_name,
                    leads=["MLII", "V1"],  # Derivações padrão MIT-BIH
                    metadata={
                        "dataset": "mit-bih",
                        "record_name": record_name,
                        "duration": len(signal_data) / record.fs,
                        "units": record.units,
                    },
                    annotations={
                        "sample": annotation.sample,
                        "symbol": annotation.symbol,
                        "aux_note": annotation.aux_note,
                    },
                )

                ecg_records.append(ecg_record)

            except Exception as e:
                self.logger.warning(f"Erro ao carregar {header_file.stem}: {e}")
                continue

        self.logger.info(f"✓ Carregados {len(ecg_records)} registros do MIT-BIH")
        return ecg_records

    def load_ptb_xl(
        self,
        dataset_path: str,
        sampling_rate: int = 100,
        max_records: int | None = None,
        preprocess: bool = True,
    ) -> list[ECGRecord]:
        """
        Carrega registros do PTB-XL

        Args:
            dataset_path: Caminho para o dataset
            sampling_rate: Taxa de amostragem desejada (100 ou 500 Hz)
            max_records: Número máximo de registros
            preprocess: Se True, aplica pré-processamento
        """
        dataset_path_obj = Path(dataset_path)

        extracted_dir = None
        for item in dataset_path_obj.iterdir():
            if item.is_dir() and "ptb-xl" in item.name.lower():
                extracted_dir = item
                break

        if not extracted_dir:
            self.logger.error("Diretório PTB-XL não encontrado")
            return []

        metadata_file = extracted_dir / "ptbxl_database.csv"
        if not metadata_file.exists():
            self.logger.error("Arquivo de metadados não encontrado")
            return []

        self.logger.info("Carregando metadados do PTB-XL...")
        metadata_df = pd.read_csv(metadata_file)

        if max_records:
            metadata_df = metadata_df.head(max_records)

        records_dir = extracted_dir / f"records{sampling_rate}"
        if not records_dir.exists():
            self.logger.error(f"Diretório de registros não encontrado: {records_dir}")
            return []

        ecg_records = []

        for idx, (_, row) in enumerate(
            tqdm(
                metadata_df.iterrows(), total=len(metadata_df), desc="Carregando PTB-XL"
            )
        ):
            try:
                filename_lr = (
                    row["filename_lr"] if sampling_rate == 100 else row["filename_hr"]
                )
                record_path = records_dir / filename_lr

                if not record_path.exists():
                    continue

                record = wfdb.rdrecord(str(record_path.with_suffix("")))
                signal_data = record.p_signal

                if preprocess and self.preprocessor:
                    try:
                        processed_signal, quality_metrics = (
                            self.preprocessor.advanced_preprocessing_pipeline(
                                signal_data[:, 0], clinical_mode=True
                            )
                        )

                        if quality_metrics["quality_score"] > 0.5:
                            signal_data[:, 0] = processed_signal[: len(signal_data)]

                    except Exception as e:
                        self.logger.warning(f"Erro no pré-processamento: {e}")

                scp_codes = eval(row["scp_codes"]) if pd.notna(row["scp_codes"]) else {}
                labels = list(scp_codes.keys())

                ecg_record = ECGRecord(
                    signal=signal_data,
                    sampling_rate=record.fs,
                    labels=labels,
                    patient_id=str(row["ecg_id"]),
                    age=row.get("age"),
                    sex=row.get("sex"),
                    leads=[
                        "I",
                        "II",
                        "III",
                        "aVR",
                        "aVL",
                        "aVF",
                        "V1",
                        "V2",
                        "V3",
                        "V4",
                        "V5",
                        "V6",
                    ],
                    metadata={
                        "dataset": "ptb-xl",
                        "ecg_id": row["ecg_id"],
                        "recording_date": row["recording_date"],
                        "height": row.get("height"),
                        "weight": row.get("weight"),
                        "nurse_validated": row.get("validated_by_human"),
                        "baseline_drift": row.get("baseline_drift"),
                        "static_noise": row.get("static_noise"),
                        "burst_noise": row.get("burst_noise"),
                        "electrodes_problems": row.get("electrodes_problems"),
                        "report": row.get("report"),
                    },
                )

                ecg_records.append(ecg_record)

            except Exception as e:
                if idx < 5:  # Mostrar apenas primeiros erros
                    self.logger.warning(f"Erro ao carregar registro {idx}: {e}")
                continue

        self.logger.info(f"✓ Carregados {len(ecg_records)} registros do PTB-XL")
        return ecg_records

    def load_physionet_challenge_2020(
        self, dataset_path: str, max_records: int | None = None, preprocess: bool = True
    ) -> list[ECGRecord]:
        """
        Carrega registros do PhysioNet Challenge 2020

        Args:
            dataset_path: Caminho para o dataset
            max_records: Número máximo de registros
            preprocess: Se True, aplica pré-processamento
        """
        dataset_path_obj = Path(dataset_path)

        extracted_dir = None
        for item in dataset_path_obj.iterdir():
            if item.is_dir() and "challenge-2020" in item.name.lower():
                extracted_dir = item
                break

        if not extracted_dir:
            self.logger.error("Diretório PhysioNet Challenge 2020 não encontrado")
            return []

        data_files = list(extracted_dir.glob("**/*.mat")) + list(
            extracted_dir.glob("**/*.hea")
        )

        if not data_files:
            self.logger.error("Arquivos de dados não encontrados")
            return []

        if max_records:
            data_files = data_files[:max_records]

        ecg_records = []

        for data_file in tqdm(data_files, desc="Carregando PhysioNet Challenge 2020"):
            try:
                if data_file.suffix == ".hea":
                    record = wfdb.rdrecord(str(data_file.with_suffix("")))
                    signal_data = record.p_signal

                    if preprocess and self.preprocessor:
                        try:
                            processed_signal, quality_metrics = (
                                self.preprocessor.advanced_preprocessing_pipeline(
                                    signal_data[:, 0], clinical_mode=True
                                )
                            )

                            if quality_metrics["quality_score"] > 0.5:
                                signal_data[:, 0] = processed_signal[: len(signal_data)]
                        except Exception as e:
                            self.logger.warning(f"Erro no pré-processamento: {e}")

                    ecg_record = ECGRecord(
                        signal=signal_data,
                        sampling_rate=record.fs,
                        labels=["challenge_2020"],  # Labels específicos do challenge
                        patient_id=data_file.stem,
                        leads=[
                            "I",
                            "II",
                            "III",
                            "aVR",
                            "aVL",
                            "aVF",
                            "V1",
                            "V2",
                            "V3",
                            "V4",
                            "V5",
                            "V6",
                        ],
                        metadata={
                            "dataset": "physionet-challenge-2020",
                            "record_name": data_file.stem,
                            "duration": len(signal_data) / record.fs,
                            "units": record.units if hasattr(record, "units") else None,
                        },
                    )

                    ecg_records.append(ecg_record)

            except Exception as e:
                self.logger.warning(f"Erro ao carregar {data_file.stem}: {e}")
                continue

        self.logger.info(
            f"✓ Carregados {len(ecg_records)} registros do PhysioNet Challenge 2020"
        )
        return ecg_records

    def load_physionet_challenge_2021(
        self, dataset_path: str, max_records: int | None = None, preprocess: bool = True
    ) -> list[ECGRecord]:
        """
        Carrega registros do PhysioNet Challenge 2021

        Args:
            dataset_path: Caminho para o dataset
            max_records: Número máximo de registros
            preprocess: Se True, aplica pré-processamento
        """
        dataset_path_obj = Path(dataset_path)

        extracted_dir = None
        for item in dataset_path_obj.iterdir():
            if item.is_dir() and "challenge-2021" in item.name.lower():
                extracted_dir = item
                break

        if not extracted_dir:
            self.logger.error("Diretório PhysioNet Challenge 2021 não encontrado")
            return []

        data_files = list(extracted_dir.glob("**/*.mat")) + list(
            extracted_dir.glob("**/*.hea")
        )

        if not data_files:
            self.logger.error("Arquivos de dados não encontrados")
            return []

        if max_records:
            data_files = data_files[:max_records]

        ecg_records = []

        for data_file in tqdm(data_files, desc="Carregando PhysioNet Challenge 2021"):
            try:
                if data_file.suffix == ".hea":
                    record = wfdb.rdrecord(str(data_file.with_suffix("")))
                    signal_data = record.p_signal

                    if preprocess and self.preprocessor:
                        try:
                            processed_signal, quality_metrics = (
                                self.preprocessor.advanced_preprocessing_pipeline(
                                    signal_data[:, 0], clinical_mode=True
                                )
                            )

                            if quality_metrics["quality_score"] > 0.5:
                                signal_data[:, 0] = processed_signal[: len(signal_data)]
                        except Exception as e:
                            self.logger.warning(f"Erro no pré-processamento: {e}")

                    ecg_record = ECGRecord(
                        signal=signal_data,
                        sampling_rate=record.fs,
                        labels=["challenge_2021"],  # Labels específicos do challenge
                        patient_id=data_file.stem,
                        leads=[
                            "I",
                            "II",
                            "III",
                            "aVR",
                            "aVL",
                            "aVF",
                            "V1",
                            "V2",
                            "V3",
                            "V4",
                            "V5",
                            "V6",
                        ],
                        metadata={
                            "dataset": "physionet-challenge-2021",
                            "record_name": data_file.stem,
                            "duration": len(signal_data) / record.fs,
                            "units": record.units if hasattr(record, "units") else None,
                        },
                    )

                    ecg_records.append(ecg_record)

            except Exception as e:
                self.logger.warning(f"Erro ao carregar {data_file.stem}: {e}")
                continue

        self.logger.info(
            f"✓ Carregados {len(ecg_records)} registros do PhysioNet Challenge 2021"
        )
        return ecg_records

    def load_mimic_iv_ecg(
        self, dataset_path: str, max_records: int | None = None, preprocess: bool = True
    ) -> list[ECGRecord]:
        """
        Carrega registros do MIMIC-IV-ECG

        Args:
            dataset_path: Caminho para o dataset
            max_records: Número máximo de registros
            preprocess: Se True, aplica pré-processamento
        """
        dataset_path_obj = Path(dataset_path)

        if not dataset_path_obj.exists():
            self.logger.error(f"Dataset MIMIC-IV-ECG não encontrado: {dataset_path}")
            return []

        data_files = list(dataset_path_obj.glob("**/*.hea"))

        if not data_files:
            self.logger.error("Arquivos de dados MIMIC-IV-ECG não encontrados")
            return []

        if max_records:
            data_files = data_files[:max_records]

        ecg_records = []

        for data_file in tqdm(data_files, desc="Carregando MIMIC-IV-ECG"):
            try:
                record = wfdb.rdrecord(str(data_file.with_suffix("")))
                signal_data = record.p_signal

                if preprocess and self.preprocessor:
                    try:
                        processed_signal, quality_metrics = (
                            self.preprocessor.advanced_preprocessing_pipeline(
                                signal_data[:, 0], clinical_mode=True
                            )
                        )

                        if quality_metrics["quality_score"] > 0.5:
                            signal_data[:, 0] = processed_signal[: len(signal_data)]
                    except Exception as e:
                        self.logger.warning(f"Erro no pré-processamento: {e}")

                ecg_record = ECGRecord(
                    signal=signal_data,
                    sampling_rate=record.fs,
                    labels=["mimic_icu"],  # Labels específicos do MIMIC
                    patient_id=data_file.stem,
                    leads=[
                        "I",
                        "II",
                        "III",
                        "aVR",
                        "aVL",
                        "aVF",
                        "V1",
                        "V2",
                        "V3",
                        "V4",
                        "V5",
                        "V6",
                    ],
                    metadata={
                        "dataset": "mimic-iv-ecg",
                        "record_name": data_file.stem,
                        "duration": len(signal_data) / record.fs,
                        "units": record.units if hasattr(record, "units") else None,
                    },
                )

                ecg_records.append(ecg_record)

            except Exception as e:
                self.logger.warning(f"Erro ao carregar {data_file.stem}: {e}")
                continue

        self.logger.info(f"✓ Carregados {len(ecg_records)} registros do MIMIC-IV-ECG")
        return ecg_records

    def load_icentia11k(
        self, dataset_path: str, max_records: int | None = None, preprocess: bool = True
    ) -> list[ECGRecord]:
        """
        Carrega registros do Icentia11k

        Args:
            dataset_path: Caminho para o dataset
            max_records: Número máximo de registros
            preprocess: Se True, aplica pré-processamento
        """
        dataset_path_obj = Path(dataset_path)

        if not dataset_path_obj.exists():
            self.logger.error(f"Dataset Icentia11k não encontrado: {dataset_path}")
            return []

        data_files = list(dataset_path_obj.glob("**/*.hdf5")) + list(
            dataset_path_obj.glob("**/*.h5")
        )

        if not data_files:
            self.logger.error("Arquivos de dados Icentia11k não encontrados")
            return []

        if max_records:
            data_files = data_files[:max_records]

        ecg_records = []

        for data_file in tqdm(data_files, desc="Carregando Icentia11k"):
            try:
                with h5py.File(data_file, "r") as f:
                    if "ecg" in f:
                        signal_data = f["ecg"][:]
                        sampling_rate = f.attrs.get("sampling_rate", 250)

                        if signal_data.ndim == 1:
                            signal_data = signal_data.reshape(-1, 1)

                        if preprocess and self.preprocessor:
                            try:
                                processed_signal, quality_metrics = (
                                    self.preprocessor.advanced_preprocessing_pipeline(
                                        signal_data[:, 0], clinical_mode=True
                                    )
                                )

                                if quality_metrics["quality_score"] > 0.5:
                                    signal_data[:, 0] = processed_signal[
                                        : len(signal_data)
                                    ]
                            except Exception as e:
                                self.logger.warning(f"Erro no pré-processamento: {e}")

                        ecg_record = ECGRecord(
                            signal=signal_data,
                            sampling_rate=int(sampling_rate),
                            labels=[
                                "continuous_monitoring"
                            ],  # Labels específicos do Icentia11k
                            patient_id=data_file.stem,
                            leads=["Lead_I"],  # Single lead
                            metadata={
                                "dataset": "icentia11k",
                                "record_name": data_file.stem,
                                "duration": len(signal_data) / sampling_rate,
                                "continuous_monitoring": True,
                            },
                        )

                        ecg_records.append(ecg_record)

            except Exception as e:
                self.logger.warning(f"Erro ao carregar {data_file.stem}: {e}")
                continue

        self.logger.info(f"✓ Carregados {len(ecg_records)} registros do Icentia11k")
        return ecg_records

    def create_unified_dataset(
        self,
        datasets: dict[str, list[ECGRecord]],
        output_path: str = "unified_ecg_dataset.h5",
    ) -> str:
        """
        Cria um dataset unificado em formato HDF5

        Args:
            datasets: Dicionário com nome do dataset -> lista de ECGRecords
            output_path: Caminho para salvar o arquivo HDF5
        """
        self.logger.info("Criando Dataset Unificado...")

        with h5py.File(output_path, "w") as hf:
            total_records = 0

            for dataset_name, records in datasets.items():
                self.logger.info(f"Processando {dataset_name}...")

                dataset_group = hf.create_group(dataset_name)

                for i, record in enumerate(
                    tqdm(records, desc=f"Salvando {dataset_name}")
                ):
                    record_group = dataset_group.create_group(f"record_{i:05d}")

                    record_group.create_dataset(
                        "signal", data=record.signal, compression="gzip"
                    )

                    record_group.attrs["sampling_rate"] = record.sampling_rate
                    record_group.attrs["patient_id"] = record.patient_id
                    record_group.attrs["labels"] = json.dumps(record.labels)

                    if record.age is not None:
                        record_group.attrs["age"] = record.age
                    if record.sex is not None:
                        record_group.attrs["sex"] = record.sex
                    if record.leads:
                        record_group.attrs["leads"] = json.dumps(record.leads)
                    if record.metadata:
                        record_group.attrs["metadata"] = json.dumps(record.metadata)

                    total_records += 1

                dataset_group.attrs["num_records"] = len(records)

            hf.attrs["total_records"] = total_records
            hf.attrs["creation_date"] = datetime.now().isoformat()
            hf.attrs["datasets"] = json.dumps(list(datasets.keys()))

        self.logger.info(f"✓ Dataset unificado criado com {total_records} registros")
        self.logger.info(f"✓ Salvo em: {output_path}")

        return output_path


class ECGDatasetAnalyzer:
    """Analisador estatístico para datasets de ECG"""

    def __init__(self) -> None:
        self.stats: dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)

    def analyze_dataset(
        self, records: list[ECGRecord], dataset_name: str = "Dataset"
    ) -> dict[str, Any]:
        """
        Analisa estatísticas de um dataset

        Args:
            records: Lista de ECGRecords
            dataset_name: Nome do dataset para identificação
        """
        self.logger.info(f"Analisando {dataset_name}...")

        stats: dict[str, Any] = {
            "dataset_name": dataset_name,
            "total_records": len(records),
            "total_duration_hours": 0.0,
            "sampling_rates": {},
            "lead_counts": {},
            "label_distribution": {},
            "age_distribution": {"mean": None, "std": None, "min": None, "max": None},
            "sex_distribution": {},
            "signal_quality": {"mean": None, "std": None},
            "average_length_seconds": 0.0,
        }

        ages = []
        durations = []

        for record in tqdm(records, desc="Analisando registros"):
            duration = len(record.signal) / record.sampling_rate
            durations.append(duration)
            stats["total_duration_hours"] = (
                float(stats["total_duration_hours"]) + duration / 3600
            )

            fs = record.sampling_rate
            sampling_rates = stats["sampling_rates"]
            sampling_rates[fs] = sampling_rates.get(fs, 0) + 1

            n_leads = len(record.leads) if record.leads else 1
            lead_counts = stats["lead_counts"]
            lead_counts[n_leads] = lead_counts.get(n_leads, 0) + 1

            for label in record.labels:
                label_distribution = stats["label_distribution"]
                label_distribution[label] = label_distribution.get(label, 0) + 1

            if record.age is not None:
                ages.append(record.age)

            if record.sex:
                sex_distribution = stats["sex_distribution"]
                sex_distribution[record.sex] = sex_distribution.get(record.sex, 0) + 1

        stats["average_length_seconds"] = np.mean(durations)

        if ages:
            stats["age_distribution"] = {
                "mean": np.mean(ages),
                "std": np.std(ages),
                "min": np.min(ages),
                "max": np.max(ages),
            }

        self._print_summary(stats)

        self.stats[dataset_name] = stats
        return stats

    def _print_summary(self, stats: dict[str, Any]) -> None:
        """Imprime resumo das estatísticas"""
        print(f"\nTotal de registros: {stats['total_records']}")
        print(f"Duração total: {stats['total_duration_hours']:.1f} horas")
        print(f"Duração média: {stats['average_length_seconds']:.1f} segundos")

        print("\nTaxas de amostragem:")
        for fs, count in stats["sampling_rates"].items():
            print(
                f"  {fs} Hz: {count} registros ({count/stats['total_records']*100:.1f}%)"
            )

        print("\nDistribuição de labels:")
        sorted_labels = sorted(
            stats["label_distribution"].items(), key=lambda x: x[1], reverse=True
        )
        for label, count in sorted_labels[:10]:  # Top 10
            print(f"  {label}: {count} ({count/stats['total_records']*100:.1f}%)")

        if stats["age_distribution"]["mean"]:
            print(
                f"\nIdade: {stats['age_distribution']['mean']:.1f} ± {stats['age_distribution']['std']:.1f} anos"
            )
            print(
                f"  Range: {stats['age_distribution']['min']}-{stats['age_distribution']['max']} anos"
            )

        if stats["sex_distribution"]:
            print("\nDistribuição por sexo:")
            for sex, count in stats["sex_distribution"].items():
                print(f"  {sex}: {count} ({count/stats['total_records']*100:.1f}%)")


def quick_download_datasets(
    datasets: list[str] | None = None, base_dir: str = "ecg_datasets"
) -> dict[str, str]:
    """
    Download rápido de datasets

    Args:
        datasets: Lista de datasets para baixar
        base_dir: Diretório base para salvar

    Returns:
        Dicionário com caminhos dos datasets baixados
    """
    if datasets is None:
        datasets = ["mit-bih"]

    downloader = ECGDatasetDownloader(base_dir)
    paths = {}

    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Baixando {dataset}...")
        print("=" * 50)

        if dataset == "mit-bih":
            path = downloader.download_mit_bih()
            if path:
                paths[dataset] = path
        elif dataset == "ptb-xl":
            path = downloader.download_ptb_xl()
            if path:
                paths[dataset] = path
        elif dataset == "cpsc2018":
            path = downloader.download_cpsc2018()
            if path:
                paths[dataset] = path
        else:
            print(f"Dataset '{dataset}' não reconhecido")

    return paths


def load_and_preprocess_all(
    dataset_paths: dict[str, str], max_records_per_dataset: int | None = None
) -> dict[str, list[ECGRecord]]:
    """
    Carrega e pré-processa todos os datasets

    Args:
        dataset_paths: Dicionário com nome_dataset -> caminho
        max_records_per_dataset: Limite de registros por dataset

    Returns:
        Dicionário com dados carregados
    """
    loader = ECGDatasetLoader()
    all_datasets = {}

    for dataset_name, path in dataset_paths.items():
        print(f"\n{'='*50}")
        print(f"Carregando {dataset_name}...")
        print("=" * 50)

        if dataset_name == "mit-bih":
            records = loader.load_mit_bih(path, preprocess=True)
        elif dataset_name == "ptb-xl":
            records = loader.load_ptb_xl(
                path, max_records=max_records_per_dataset, preprocess=True
            )
        else:
            print(f"Loader não implementado para {dataset_name}")
            continue

        if records:
            all_datasets[dataset_name] = records

    return all_datasets


def prepare_ml_dataset(
    records: list[ECGRecord],
    window_size: int = 3600,
    target_labels: list[str] | None = None,
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """
    Prepara dataset para treinamento de ML

    Args:
        records: Lista de ECGRecords
        window_size: Tamanho da janela em amostras
        target_labels: Labels específicos para filtrar (None = todos)

    Returns:
        X: Array de sinais (n_samples, window_size)
        y: Array de labels (n_samples,)
    """
    X: list[np.ndarray[Any, Any]] = []
    y: list[int] = []

    if target_labels is None:
        all_labels = set()
        for record in records:
            all_labels.update(record.labels)
        target_labels = sorted(all_labels)

    label_to_idx = {label: idx for idx, label in enumerate(target_labels)}

    print(f"\nPreparando dataset com {len(target_labels)} classes")

    for record in tqdm(records, desc="Preparando dados"):
        signal = record.signal
        if len(signal.shape) > 1:
            signal = signal[:, 0]  # Usar primeira derivação se multi-canal

        for i in range(0, len(signal) - window_size, window_size // 2):
            window = signal[i : i + window_size]

            window_label = None
            for label in record.labels:
                if label in label_to_idx:
                    window_label = label_to_idx[label]
                    break

            if window_label is not None:
                X.append(window)
                y.append(window_label)

    X_array = np.array(X)
    y_array = np.array(y)

    print(
        f"✓ Dataset preparado: {X_array.shape[0]} amostras de shape {X_array.shape[1:]}"
    )
    print("✓ Distribuição de classes:")

    for label, idx in label_to_idx.items():
        count = np.sum(y_array == idx)
        print(f"   {label}: {count} ({count/len(y_array)*100:.1f}%)")

    return X_array, y_array
