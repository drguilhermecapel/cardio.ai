# backend/training/dataset_configs.py

from dataclasses import dataclass
import os

@dataclass
class DatasetConfig:
    name: str
    description: str
    download_size: str
    url: str
    classes: list
    sampling_rate: int
    num_leads: int

DATASET_CONFIGS = {
    "ptbxl": DatasetConfig(
        name="PTB-XL",
        description="A large publicly available electrocardiography dataset",
        download_size="~3 GB",
        url="https://physionet.org/content/ptb-xl/1.0.3/",
        classes=["NORM", "MI", "STTC", "CD", "HYP"],
        sampling_rate=100,
        num_leads=12
    ),
    "mitbih": DatasetConfig(
        name="MIT-BIH Arrhythmia Database",
        description="Recordings of two-channel ambulatory ECGs",
        download_size="~100 MB",
        url="https://physionet.org/content/mitdb/1.0.0/",
        classes=["N", "SVEB", "VEB", "F", "Q"],
        sampling_rate=360,
        num_leads=2
    ),
    "cpsc2018": DatasetConfig(
        name="CPSC2018",
        description="China Physiological Signal Challenge 2018",
        download_size="~2 GB",
        url="http://2018.challenge.physionet.org/",
        classes=["Normal", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"],
        sampling_rate=500,
        num_leads=12
    )
}

DOWNLOAD_LINKS = {
    "ptbxl": {
        "data": "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip",
        "metadata": "https://physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"
    },
    "mitbih": {
        "data": "https://physionet.org/files/mitdb/1.0.0/",
        "instructions": "Use wfdb.dl_database(\'mitdb\', ...) to download"
    },
    "cpsc2018": {
        "data": "http://2018.challenge.physionet.org/",
        "training": "http://2018.challenge.physionet.org/TrainingSet.zip",
        "test": "http://2018.challenge.physionet.org/TestSet.zip",
        "instructions": "Download manually from the challenge website"
    }
}

def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Retorna a configuração de um dataset específico."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset {dataset_name} não suportado.")
    return DATASET_CONFIGS[dataset_name]


