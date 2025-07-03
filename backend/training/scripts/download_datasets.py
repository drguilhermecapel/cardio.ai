"""
Script aprimorado para download de datasets públicos de ECG
"""

import requests
import zipfile
import tarfile
import gzip
import shutil
from pathlib import Path
import logging
from typing import Dict, Any
import argparse
import urllib.request
import sys

# Adicionar o diretório pai (training) ao path para permitir importações relativas
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.dataset_configs import DATASET_CONFIGS, DOWNLOAD_LINKS
from config.training_config import training_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Classe para download automático de datasets de ECG"""
    
    def __init__(self, data_root: Path = None):
        self.data_root = data_root or training_config.DATA_ROOT
        self.data_root.mkdir(parents=True, exist_ok=True)
        
    def download_progress_hook(self, block_num, block_size, total_size):
        """Hook para mostrar progresso do download"""
        downloaded = block_num * block_size
        percent = min(downloaded * 100 / total_size, 100) if total_size > 0 else 0
        bar_length = 40
        filled_length = int(bar_length * percent // 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        sys.stdout.write(f'\r|{bar}| {percent:.1f}% ({downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB)')
        sys.stdout.flush()
        
    def download_ptbxl(self, force_download: bool = False):
        """Download específico do PTB-XL com progresso"""
        dataset_path = self.data_root / "ptbxl"
        dataset_path.mkdir(exist_ok=True)
        
        # Verificar se os dados já existem e se não é para forçar o download
        if (dataset_path / "ptbxl_database.csv").exists() and not force_download:
            logger.info("PTB-XL já existe. Use --force para redownload.")
            return dataset_path
            
        logger.info("Baixando PTB-XL Database (~3GB)...")
        
        zip_url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"
        zip_path = dataset_path / "ptb-xl.zip"
        
        try:
            # Download do arquivo ZIP se não existir ou se for forçado
            if not zip_path.exists() or force_download:
                logger.info(f"Baixando de {zip_url}...")
                urllib.request.urlretrieve(
                    zip_url, 
                    zip_path,
                    reporthook=self.download_progress_hook
                )
                print()  # Nova linha após progresso
            
            # Extrair
            logger.info("Extraindo arquivos...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_path)
                
            # Mover arquivos da pasta extraída
            extracted = dataset_path / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
            if extracted.exists():
                for item in extracted.iterdir():
                    shutil.move(str(item), str(dataset_path / item.name))
                extracted.rmdir()
                
            # Remover ZIP
            zip_path.unlink()
            
            logger.info(f"✓ PTB-XL baixado com sucesso em: {dataset_path}")
            return dataset_path
            
        except Exception as e:
            logger.error(f"Erro ao baixar PTB-XL: {e}")
            return None
            
    def download_mitbih(self, force_download: bool = False):
        """Download do MIT-BIH usando wfdb"""
        dataset_path = self.data_root / "mitbih"
        dataset_path.mkdir(exist_ok=True)
        
        # Verificar se já existe
        if len(list(dataset_path.glob("*.dat"))) > 0 and not force_download:
            logger.info("MIT-BIH já existe. Use --force para redownload.")
            return dataset_path
            
        logger.info("Baixando MIT-BIH Arrhythmia Database...")
        
        try:
            import wfdb
            
            # Lista de registros MIT-BIH
            records = [
                '100', '101', '102', '103', '104', '105', '106', '107', '108', '109',
                '111', '112', '113', '114', '115', '116', '117', '118', '119', '121',
                '122', '123', '124', '200', '201', '202', '203', '205', '207', '208',
                '209', '210', '212', '213', '214', '215', '217', '219', '220', '221',
                '222', '223', '228', '230', '231', '232', '233', '234'
            ]
            
            # Baixar cada registro
            for i, record in enumerate(records):
                logger.info(f"Baixando registro {record} ({i+1}/{len(records)})...")
                try:
                    wfdb.dl_database('mitdb', str(dataset_path), records=[record])
                except Exception as e:
                    logger.warning(f"Erro ao baixar {record}: {e}")
                    
            logger.info(f"✓ MIT-BIH baixado com sucesso em: {dataset_path}")
            return dataset_path
            
        except ImportError:
            logger.error("wfdb não está instalado. Execute: pip install wfdb")
            return None
        except Exception as e:
            logger.error(f"Erro ao baixar MIT-BIH: {e}")
            return None
            
    def download_dataset(self, dataset_name: str, force_download: bool = False):
        """Download de um dataset específico"""
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"Dataset {dataset_name} não encontrado")
            
        dataset_config = DATASET_CONFIGS[dataset_name]
        
        logger.info(f"Dataset: {dataset_config.name}")
        logger.info(f"Descrição: {dataset_config.description}")
        logger.info(f"Tamanho: {dataset_config.download_size}")
        logger.info(f"URL: {dataset_config.url}")
        
        # Métodos específicos para cada dataset
        if dataset_name == "ptbxl":
            return self.download_ptbxl(force_download)
        elif dataset_name == "mitbih":
            return self.download_mitbih(force_download)
        else:
            logger.warning(f"Download automático não implementado para {dataset_name}")
            logger.info(f"Por favor, baixe manualmente de: {dataset_config.url}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Download de datasets de ECG")
    parser.add_argument("--dataset", type=str, default="all",
                        help="Dataset para baixar (ptbxl, mitbih, all)")
    parser.add_argument("--force", action="store_true",
                        help="Forçar redownload mesmo se já existe")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader()
    
    if args.dataset == "all":
        datasets = ["ptbxl", "mitbih"]
    else:
        datasets = [args.dataset]
        
    logger.info("=" * 60)
    logger.info("DOWNLOAD DE DATASETS - CARDIOAI PRO")
    logger.info("=" * 60)
    
    for dataset in datasets:
        logger.info(f"\nBaixando {dataset}...")
        result = downloader.download_dataset(dataset, args.force)
        if result:
            logger.info(f"✅ {dataset} baixado com sucesso!")
        else:
            logger.error(f"❌ Falha ao baixar {dataset}")
            
    logger.info("\n" + "=" * 60)
    logger.info("Download concluído!")
    logger.info("Para treinar, execute:")
    logger.info("python backend/training/main.py --dataset ptbxl --model cnn_lstm")


if __name__ == "__main__":
    main()

