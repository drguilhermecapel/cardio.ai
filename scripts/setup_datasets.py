#!/usr/bin/env python3
"""
Script para configurar todos os datasets de ECG para o CardioAI Pro
Implementa os próximos passos solicitados pelo usuário
"""

import logging
import sys
from pathlib import Path

backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from app.datasets.ecg_public_datasets import (
    ECGDatasetDownloader,
    ECGDatasetLoader,
    ECGDatasetAnalyzer,
)


def setup_logging():
    """Configura logging para o script"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("dataset_setup.log")],
    )


def main():
    """Função principal para configurar todos os datasets"""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("🚀 Iniciando configuração dos datasets de ECG para CardioAI Pro")

    datasets_dir = Path("ecg_datasets")
    datasets_dir.mkdir(exist_ok=True)

    downloader = ECGDatasetDownloader(base_dir=str(datasets_dir))

    logger.info("📊 Datasets disponíveis:")
    downloader.list_available_datasets()

    logger.info("\n" + "=" * 60)
    logger.info("INICIANDO DOWNLOADS DOS DATASETS")
    logger.info("=" * 60)

    download_results = downloader.download_all_datasets(force_redownload=False)

    successful = [name for name, path in download_results.items() if path is not None]
    failed = [name for name, path in download_results.items() if path is None]

    logger.info(f"\n✅ Downloads concluídos: {len(successful)}/{len(download_results)}")
    if successful:
        logger.info(f"   Sucessos: {', '.join(successful)}")
    if failed:
        logger.warning(f"   Falhas: {', '.join(failed)}")

    logger.info("\n" + "=" * 60)
    logger.info("CARREGANDO E ANALISANDO DATASETS")
    logger.info("=" * 60)

    loader = ECGDatasetLoader()
    analyzer = ECGDatasetAnalyzer()

    loaded_datasets = {}

    if download_results.get("mit-bih"):
        try:
            logger.info("Carregando MIT-BIH...")
            mitbih_records = loader.load_mit_bih(
                download_results["mit-bih"],
                max_records=100,  # Limitar para teste
                preprocess=True,
            )
            loaded_datasets["mit-bih"] = mitbih_records

            stats = analyzer.analyze_dataset(mitbih_records, "MIT-BIH")
            logger.info(f"MIT-BIH: {stats['total_records']} registros carregados")

        except Exception as e:
            logger.error(f"Erro ao carregar MIT-BIH: {e}")

    if download_results.get("ptb-xl"):
        try:
            logger.info("Carregando PTB-XL...")
            ptbxl_records = loader.load_ptb_xl(
                download_results["ptb-xl"],
                max_records=100,  # Limitar para teste
                preprocess=True,
            )
            loaded_datasets["ptb-xl"] = ptbxl_records

            stats = analyzer.analyze_dataset(ptbxl_records, "PTB-XL")
            logger.info(f"PTB-XL: {stats['total_records']} registros carregados")

        except Exception as e:
            logger.error(f"Erro ao carregar PTB-XL: {e}")

    if download_results.get("physionet-challenge-2020"):
        try:
            logger.info("Carregando PhysioNet Challenge 2020...")
            challenge2020_records = loader.load_physionet_challenge_2020(
                download_results["physionet-challenge-2020"],
                max_records=50,  # Limitar para teste
                preprocess=True,
            )
            loaded_datasets["physionet-challenge-2020"] = challenge2020_records

            stats = analyzer.analyze_dataset(
                challenge2020_records, "PhysioNet Challenge 2020"
            )
            logger.info(
                f"PhysioNet Challenge 2020: {stats['total_records']} registros carregados"
            )

        except Exception as e:
            logger.error(f"Erro ao carregar PhysioNet Challenge 2020: {e}")

    if download_results.get("physionet-challenge-2021"):
        try:
            logger.info("Carregando PhysioNet Challenge 2021...")
            challenge2021_records = loader.load_physionet_challenge_2021(
                download_results["physionet-challenge-2021"],
                max_records=50,  # Limitar para teste
                preprocess=True,
            )
            loaded_datasets["physionet-challenge-2021"] = challenge2021_records

            stats = analyzer.analyze_dataset(
                challenge2021_records, "PhysioNet Challenge 2021"
            )
            logger.info(
                f"PhysioNet Challenge 2021: {stats['total_records']} registros carregados"
            )

        except Exception as e:
            logger.error(f"Erro ao carregar PhysioNet Challenge 2021: {e}")

    if loaded_datasets:
        logger.info("\n" + "=" * 60)
        logger.info("CRIANDO DATASET UNIFICADO")
        logger.info("=" * 60)

        try:
            unified_path = loader.create_unified_dataset(
                loaded_datasets, output_path="unified_ecg_dataset.h5"
            )
            logger.info(f"✅ Dataset unificado criado: {unified_path}")

        except Exception as e:
            logger.error(f"Erro ao criar dataset unificado: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("RELATÓRIO FINAL")
    logger.info("=" * 60)

    total_records = sum(len(records) for records in loaded_datasets.values())
    logger.info(f"📊 Total de registros carregados: {total_records}")
    logger.info(f"📁 Datasets processados: {len(loaded_datasets)}")

    for dataset_name, records in loaded_datasets.items():
        logger.info(f"   {dataset_name}: {len(records)} registros")

    logger.info("\n🎯 Próximos passos:")
    logger.info("1. ✅ Download dos datasets - CONCLUÍDO")
    logger.info("2. ⏳ Setup de GPU - PRÓXIMO")
    logger.info("3. ⏳ Treinamento do modelo híbrido - PRÓXIMO")
    logger.info("4. ⏳ Validação e otimização - PRÓXIMO")

    logger.info("\n🚀 Setup dos datasets concluído com sucesso!")


if __name__ == "__main__":
    main()
