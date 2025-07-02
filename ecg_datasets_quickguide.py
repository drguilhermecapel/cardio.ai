"""
Guia Rápido: Integração Completa com Datasets Públicos de ECG
Integrado com o sistema CardioAI Pro
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .ecg_public_datasets import (
    ECGDatasetAnalyzer,
    ECGDatasetDownloader,
    ECGDatasetLoader,
    ECGRecord,
    load_and_preprocess_all,
    prepare_ml_dataset,
    quick_download_datasets,
)


def setup_environment() -> bool:
    """Configura ambiente e verifica dependências"""
    print("🔧 Configurando ambiente...")

    dirs = ["ecg_datasets", "models", "results", "figures"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)

    try:
        print("✓ wfdb instalado")
    except ImportError:
        print("✗ wfdb não instalado. Execute: pip install wfdb")
        return False

    try:
        print("✓ h5py instalado")
    except ImportError:
        print("✗ h5py não instalado. Execute: pip install h5py")
        return False

    print("✓ Ambiente configurado com sucesso!\n")
    return True


def scenario_1_download_and_explore() -> list[ECGRecord] | None:
    """Cenário 1: Baixar datasets e fazer análise exploratória"""

    print("\n" + "=" * 60)
    print("CENÁRIO 1: Download e Exploração de Datasets")
    print("=" * 60)

    downloader = ECGDatasetDownloader()

    print("\n1️⃣ Baixando MIT-BIH (5 registros para teste rápido)...")
    mit_path = downloader.download_mit_bih(
        records_to_download=["100", "101", "102", "103", "104"]
    )

    if not mit_path:
        print("Erro no download!")
        return None

    print("\n2️⃣ Carregando e pré-processando dados...")
    loader = ECGDatasetLoader()
    records = loader.load_mit_bih(mit_path, preprocess=True)

    print(f"✓ Carregados {len(records)} registros")

    print("\n3️⃣ Análise exploratória...")
    analyzer = ECGDatasetAnalyzer()
    analyzer.analyze_dataset(records, "MIT-BIH Sample")

    print("\n4️⃣ Gerando visualizações...")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    for i, record in enumerate(records[:4]):
        ax = axes[i]

        samples = min(10 * record.sampling_rate, len(record.signal))
        t = np.arange(samples) / record.sampling_rate

        ax.plot(t, record.signal[:samples], linewidth=0.8)
        ax.set_title(f'{record.patient_id} - Labels: {", ".join(record.labels[:3])}')
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Amplitude (mV)")
        ax.grid(True, alpha=0.3)

        ax.text(
            0.02,
            0.95,
            f"FS: {record.sampling_rate} Hz",
            transform=ax.transAxes,
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
        )

    plt.tight_layout()
    plt.savefig("figures/mit_bih_examples.png", dpi=150)
    print("✓ Visualização salva em: figures/mit_bih_examples.png")

    return records


def scenario_2_prepare_ml_dataset(
    records: list[ECGRecord] | None = None,
) -> tuple[
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    np.ndarray[Any, Any],
    list[str],
]:
    """Cenário 2: Preparar dataset para treinamento de ML"""

    print("\n" + "=" * 60)
    print("CENÁRIO 2: Preparação de Dataset para ML")
    print("=" * 60)

    if records is None:
        loader = ECGDatasetLoader()
        records = loader.load_mit_bih("ecg_datasets/mit-bih", preprocess=True)

    print("\n1️⃣ Preparando dataset para ML...")

    target_labels = ["normal", "pvc", "pac", "arrhythmia"]

    X, y = prepare_ml_dataset(
        records, window_size=3600, target_labels=target_labels  # 10 segundos @ 360Hz
    )

    print("\n2️⃣ Dividindo dados...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"✓ Treino: {X_train.shape}")
    print(f"✓ Teste: {X_test.shape}")

    print("\n3️⃣ Normalizando dados...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.T).T
    X_test_scaled = scaler.transform(X_test.T).T

    X_train_cnn = X_train_scaled.reshape(
        X_train_scaled.shape[0], X_train_scaled.shape[1], 1
    )
    X_test_cnn = X_test_scaled.reshape(
        X_test_scaled.shape[0], X_test_scaled.shape[1], 1
    )

    print("\n4️⃣ Salvando dados preparados...")
    np.savez_compressed(
        "ecg_datasets/ml_ready_mit_bih.npz",
        X_train=X_train_cnn,
        X_test=X_test_cnn,
        y_train=y_train,
        y_test=y_test,
        labels=target_labels,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
    )

    print("✓ Dataset ML salvo em: ecg_datasets/ml_ready_mit_bih.npz")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    unique, counts = np.unique(y_train, return_counts=True)
    ax1.bar(unique, counts, color="skyblue", edgecolor="black")
    ax1.set_xticks(unique)
    ax1.set_xticklabels([target_labels[i] for i in unique], rotation=45)
    ax1.set_title("Distribuição de Classes (Treino)")
    ax1.set_ylabel("Quantidade")

    for i, label in enumerate(target_labels):
        idx = np.where(y_train == i)[0][0]
        signal = X_train[idx][:1000]  # Primeiros 1000 pontos
        t = np.arange(len(signal)) / 360
        ax2.plot(t, signal + i * 2, label=label, linewidth=1)

    ax2.set_title("Exemplos de Sinais por Classe")
    ax2.set_xlabel("Tempo (s)")
    ax2.set_ylabel("Amplitude (normalizada)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/ml_dataset_overview.png", dpi=150)
    print("✓ Visualização salva em: figures/ml_dataset_overview.png")

    return X_train_cnn, X_test_cnn, y_train, y_test, target_labels


def quick_start_mit_bih() -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    """Início rápido com MIT-BIH"""
    paths = quick_download_datasets(["mit-bih"])

    datasets = load_and_preprocess_all(paths)

    X, y = prepare_ml_dataset(datasets["mit-bih"])

    print("\n✅ Pronto para ML!")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")

    return X, y


def quick_visualize_ecg(record: ECGRecord, duration: float = 10.0) -> None:
    """Visualização rápida de um ECG"""
    samples = min(int(duration * record.sampling_rate), len(record.signal))
    t = np.arange(samples) / record.sampling_rate

    plt.figure(figsize=(12, 4))

    if len(record.signal.shape) > 1:
        for i in range(min(3, record.signal.shape[1])):
            plt.plot(
                t,
                record.signal[:samples, i] + i * 2,
                label=record.leads[i] if record.leads else f"Ch{i}",
            )
    else:
        plt.plot(t, record.signal[:samples])

    plt.title(f'ECG: {record.patient_id} - {", ".join(record.labels[:3])}')
    plt.xlabel("Tempo (s)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Função principal com menu interativo"""

    print("\n" + "=" * 70)
    print("🏥 SISTEMA DE INTEGRAÇÃO COM DATASETS PÚBLICOS DE ECG")
    print("=" * 70)

    if not setup_environment():
        print("\n❌ Erro na configuração do ambiente!")
        return

    while True:
        print("\n📋 MENU DE CENÁRIOS:")
        print("1. Download e Exploração Inicial (MIT-BIH)")
        print("2. Preparação de Dataset para ML")
        print("3. Início Rápido MIT-BIH")
        print("0. Sair")

        choice = input("\nEscolha um cenário (0-3): ").strip()

        if choice == "0":
            print("\n👋 Até logo!")
            break
        elif choice == "1":
            scenario_1_download_and_explore()
        elif choice == "2":
            scenario_2_prepare_ml_dataset()
        elif choice == "3":
            X, y = quick_start_mit_bih()
        else:
            print("\n❌ Opção inválida!")

        input("\nPressione ENTER para continuar...")


if __name__ == "__main__":
    main()
