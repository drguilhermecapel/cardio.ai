#!/usr/bin/env python3
"""
Processador Científico PTB-XL para Deep Learning em Diagnóstico Cardíaco
Implementa processamento robusto com validação médica dos sinais ECG
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import logging
from typing import Dict, Tuple, List, Optional
import warnings
from scipy import signal as scipy_signal
from scipy.stats import zscore
import matplotlib.pyplot as plt

# Configuração científica
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PTBXLProcessor:
    """
    Processador científico para o dataset PTB-XL com validação médica
    """
    
    # Constantes médicas baseadas em diretrizes cardiológicas
    LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # Limites fisiológicos de amplitude (mV)
    MIN_AMPLITUDE = -5.0  # mV
    MAX_AMPLITUDE = 5.0   # mV
    
    # Parâmetros de qualidade do sinal
    MIN_SIGNAL_QUALITY = 0.7  # 70% do sinal deve estar dentro dos limites
    MAX_BASELINE_DRIFT = 0.5  # mV
    
    # Mapeamento de patologias SCP-ECG para classes
    PATHOLOGY_MAPPING = {
        'NORM': 0,  # Normal
        'MI': 1,    # Infarto do miocárdio
        'STTC': 2,  # Alterações ST-T
        'CD': 3,    # Distúrbios de condução
        'HYP': 4,   # Hipertrofia
        'AFIB': 5,  # Fibrilação atrial
        'OTHER': 6  # Outras patologias
    }
    
    def __init__(self, data_path: str, sampling_rate: int = 100):
        """
        Inicializa o processador
        
        Args:
            data_path: Caminho para os arquivos .npy
            sampling_rate: Taxa de amostragem (100 ou 500 Hz)
        """
        self.data_path = Path(data_path)
        self.sampling_rate = sampling_rate
        self.signal_length = 10 * sampling_rate  # 10 segundos
        
        # Estatísticas de processamento
        self.stats = {
            'total_files': 0,
            'processed': 0,
            'rejected_quality': 0,
            'rejected_missing': 0,
            'pathology_counts': defaultdict(int)
        }
        
    def validate_signal_quality(self, signal: np.ndarray) -> bool:
        """
        Valida a qualidade do sinal ECG segundo critérios médicos
        
        Args:
            signal: Sinal ECG de uma derivação
            
        Returns:
            bool: True se o sinal tem qualidade aceitável
        """
        # Verificar amplitude
        valid_samples = np.sum((signal >= self.MIN_AMPLITUDE) & (signal <= self.MAX_AMPLITUDE))
        quality_ratio = valid_samples / len(signal)
        
        if quality_ratio < self.MIN_SIGNAL_QUALITY:
            return False
        
        # Verificar deriva da linha de base
        baseline = scipy_signal.savgol_filter(signal, window_length=51, polyorder=3)
        baseline_drift = np.max(np.abs(baseline - np.median(baseline)))
        
        if baseline_drift > self.MAX_BASELINE_DRIFT:
            return False
        
        # Verificar ruído excessivo
        noise_level = np.std(signal - scipy_signal.savgol_filter(signal, window_length=5, polyorder=2))
        if noise_level > 0.2:  # 200 µV
            return False
        
        return True
    
    def preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Pré-processa o sinal ECG com filtros médicos padrão
        
        Args:
            signal: Sinal bruto
            
        Returns:
            Sinal processado
        """
        # Filtro passa-banda (0.5-40 Hz) - padrão para diagnóstico
        nyquist = self.sampling_rate / 2
        low_freq = 0.5 / nyquist
        high_freq = min(40.0 / nyquist, 0.99)
        
        b, a = scipy_signal.butter(4, [low_freq, high_freq], btype='band')
        filtered = scipy_signal.filtfilt(b, a, signal)
        
        # Remoção de deriva da linha de base
        baseline = scipy_signal.savgol_filter(filtered, window_length=101, polyorder=3)
        corrected = filtered - baseline
        
        # Normalização z-score
        normalized = zscore(corrected)
        
        return normalized
    
    def extract_diagnostic_features(self, ecg_12lead: np.ndarray) -> Dict[str, float]:
        """
        Extrai características diagnósticas do ECG
        
        Args:
            ecg_12lead: ECG de 12 derivações (12, N)
            
        Returns:
            Dicionário com características médicas
        """
        features = {}
        
        # Frequência cardíaca (usando derivação II)
        lead_ii = ecg_12lead[1, :]
        
        # Detector de picos R simplificado
        peaks, _ = scipy_signal.find_peaks(lead_ii, height=0.5, distance=self.sampling_rate*0.5)
        
        if len(peaks) > 1:
            rr_intervals = np.diff(peaks) / self.sampling_rate  # em segundos
            hr = 60 / np.mean(rr_intervals)
            features['heart_rate'] = hr
            features['hr_variability'] = np.std(rr_intervals) * 1000  # ms
        else:
            features['heart_rate'] = 0
            features['hr_variability'] = 0
        
        # Amplitude QRS média (mV)
        for i, lead in enumerate(self.LEAD_NAMES):
            lead_signal = ecg_12lead[i, :]
            features[f'qrs_amplitude_{lead}'] = np.max(np.abs(lead_signal))
        
        # Eixo elétrico (simplificado)
        lead_i = ecg_12lead[0, :]
        lead_avf = ecg_12lead[5, :]
        
        # Cálculo aproximado do eixo
        net_i = np.sum(lead_i)
        net_avf = np.sum(lead_avf)
        
        if net_i != 0:
            axis_angle = np.degrees(np.arctan2(net_avf, net_i))
            features['electrical_axis'] = axis_angle
        else:
            features['electrical_axis'] = 90 if net_avf > 0 else -90
        
        return features
    
    def process_dataset(self, 
                       output_path: Optional[str] = None,
                       max_samples: Optional[int] = None,
                       extract_features: bool = True) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Processa o dataset completo
        
        Args:
            output_path: Caminho para salvar os arquivos processados
            max_samples: Número máximo de amostras (para teste)
            extract_features: Se deve extrair características diagnósticas
            
        Returns:
            X: Array de sinais (N, 12, signal_length)
            Y: Array de labels (N,)
            features_df: DataFrame com características diagnósticas
        """
        # Listar arquivos
        npy_files = sorted([f for f in self.data_path.glob("*.npy") if f.stem.isdigit()])
        
        if max_samples:
            npy_files = npy_files[:max_samples]
        
        self.stats['total_files'] = len(npy_files)
        logger.info(f"Encontrados {len(npy_files)} arquivos ECG")
        
        # Verificar se existe arquivo de metadados
        metadata_file = self.data_path.parent / 'ptbxl_database.csv'
        has_metadata = metadata_file.exists()
        
        if has_metadata:
            logger.info("Carregando metadados clínicos...")
            metadata_df = pd.read_csv(metadata_file)
        else:
            logger.warning("Arquivo de metadados não encontrado. Usando labels dummy.")
        
        # Inicializar arrays
        X_list = []
        Y_list = []
        features_list = []
        valid_indices = []
        
        logger.info("Processando sinais ECG...")
        
        for idx, file_path in enumerate(tqdm(npy_files, desc="Processando ECGs")):
            try:
                # Carregar sinal
                signal = np.load(file_path)
                
                # Verificar dimensões
                if signal.ndim == 1:
                    # Assumir 12 derivações concatenadas
                    expected_length = 12 * self.signal_length
                    if len(signal) == expected_length:
                        signal = signal.reshape(12, self.signal_length)
                    else:
                        logger.warning(f"Dimensão incorreta em {file_path.name}: {signal.shape}")
                        self.stats['rejected_missing'] += 1
                        continue
                
                elif signal.shape[0] != 12 or signal.shape[1] != self.signal_length:
                    logger.warning(f"Formato incorreto em {file_path.name}: {signal.shape}")
                    self.stats['rejected_missing'] += 1
                    continue
                
                # Validar qualidade
                quality_ok = True
                for lead_idx in range(12):
                    if not self.validate_signal_quality(signal[lead_idx, :]):
                        quality_ok = False
                        break
                
                if not quality_ok:
                    self.stats['rejected_quality'] += 1
                    continue
                
                # Pré-processar
                processed_signal = np.zeros_like(signal, dtype=np.float32)
                for lead_idx in range(12):
                    processed_signal[lead_idx, :] = self.preprocess_signal(signal[lead_idx, :])
                
                # Extrair características diagnósticas
                if extract_features:
                    features = self.extract_diagnostic_features(processed_signal)
                    features['file_id'] = file_path.stem
                    features_list.append(features)
                
                # Determinar label
                if has_metadata:
                    # Buscar no metadata
                    ecg_id = int(file_path.stem)
                    meta_row = metadata_df[metadata_df['ecg_id'] == ecg_id]
                    
                    if not meta_row.empty:
                        # Simplificar para classificação multi-classe
                        scp_codes = meta_row['scp_codes'].iloc[0]
                        # Aqui você implementaria a lógica de mapeamento real
                        label = self._map_scp_to_class(scp_codes)
                    else:
                        label = 0  # Normal por padrão
                else:
                    # Label dummy
                    label = np.random.randint(0, len(self.PATHOLOGY_MAPPING))
                
                X_list.append(processed_signal)
                Y_list.append(label)
                valid_indices.append(idx)
                
                self.stats['processed'] += 1
                self.stats['pathology_counts'][label] += 1
                
            except Exception as e:
                logger.error(f"Erro processando {file_path.name}: {str(e)}")
                continue
        
        # Converter para arrays numpy
        X = np.array(X_list, dtype=np.float32)
        Y = np.array(Y_list, dtype=np.int64)
        
        # DataFrame de características
        features_df = pd.DataFrame(features_list) if features_list else None
        
        # Relatório de processamento
        self._print_processing_report()
        
        # Salvar se especificado
        if output_path:
            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Salvando dados processados em {output_dir}")
            
            np.save(output_dir / "X.npy", X)
            np.save(output_dir / "Y.npy", Y)
            
            if features_df is not None:
                features_df.to_csv(output_dir / "features.csv", index=False)
            
            # Salvar estatísticas
            with open(output_dir / "processing_stats.json", "w") as f:
                json.dump(self.stats, f, indent=2)
            
            # Criar arquivo README
            self._create_readme(output_dir, X.shape, Y.shape)
        
        return X, Y, features_df
    
    def _map_scp_to_class(self, scp_codes: str) -> int:
        """
        Mapeia códigos SCP-ECG para classes simplificadas
        
        Esta é uma implementação simplificada. Na prática, você
        usaria a lógica específica do seu problema.
        """
        # Por enquanto, retorna classe aleatória
        return np.random.randint(0, len(self.PATHOLOGY_MAPPING))
    
    def _print_processing_report(self):
        """Imprime relatório detalhado do processamento"""
        logger.info("\n" + "="*60)
        logger.info("RELATÓRIO DE PROCESSAMENTO PTB-XL")
        logger.info("="*60)
        logger.info(f"Total de arquivos: {self.stats['total_files']}")
        logger.info(f"Processados com sucesso: {self.stats['processed']}")
        logger.info(f"Rejeitados por qualidade: {self.stats['rejected_quality']}")
        logger.info(f"Rejeitados por dados faltantes: {self.stats['rejected_missing']}")
        logger.info(f"Taxa de sucesso: {self.stats['processed']/self.stats['total_files']*100:.1f}%")
        logger.info("\nDistribuição de patologias:")
        for class_id, count in sorted(self.stats['pathology_counts'].items()):
            logger.info(f"  Classe {class_id}: {count} amostras")
        logger.info("="*60)
    
    def _create_readme(self, output_dir: Path, x_shape: tuple, y_shape: tuple):
        """Cria arquivo README com informações do dataset processado"""
        readme_content = f"""# Dataset PTB-XL Processado

## Informações Gerais
- Data de processamento: {pd.Timestamp.now()}
- Taxa de amostragem: {self.sampling_rate} Hz
- Duração do sinal: 10 segundos
- Número de derivações: 12

## Dimensões dos Dados
- X (sinais): {x_shape}
- Y (labels): {y_shape}

## Pré-processamento Aplicado
1. Validação de qualidade do sinal
2. Filtro passa-banda (0.5-40 Hz)
3. Remoção de deriva da linha de base
4. Normalização z-score

## Estatísticas de Processamento
- Total de arquivos: {self.stats['total_files']}
- Processados com sucesso: {self.stats['processed']}
- Rejeitados por qualidade: {self.stats['rejected_quality']}
- Rejeitados por dados faltantes: {self.stats['rejected_missing']}

## Distribuição de Classes
{pd.DataFrame(list(self.stats['pathology_counts'].items()), 
              columns=['Classe', 'Quantidade']).to_string()}

## Critérios de Qualidade
- Amplitude: {self.MIN_AMPLITUDE} a {self.MAX_AMPLITUDE} mV
- Qualidade mínima: {self.MIN_SIGNAL_QUALITY*100}%
- Deriva máxima da baseline: {self.MAX_BASELINE_DRIFT} mV

## Uso Recomendado
```python
import numpy as np

# Carregar dados
X = np.load('X.npy')
Y = np.load('Y.npy')

# X shape: (n_samples, 12, 1000)
# Y shape: (n_samples,)
```
"""
        
        with open(output_dir / "README.md", "w") as f:
            f.write(readme_content)


def main():
    """Função principal para executar o processamento"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Processador Científico PTB-XL")
    parser.add_argument("data_path", help="Caminho para os arquivos .npy do PTB-XL")
    parser.add_argument("--output", "-o", help="Diretório de saída", default=None)
    parser.add_argument("--sampling-rate", "-s", type=int, default=100, 
                       choices=[100, 500], help="Taxa de amostragem (Hz)")
    parser.add_argument("--max-samples", "-m", type=int, default=None,
                       help="Número máximo de amostras para processar")
    parser.add_argument("--no-features", action="store_true",
                       help="Não extrair características diagnósticas")
    
    args = parser.parse_args()
    
    # Se não especificar saída, usar o mesmo diretório dos dados
    if args.output is None:
        args.output = args.data_path
    
    # Criar processador
    processor = PTBXLProcessor(args.data_path, args.sampling_rate)
    
    # Processar dataset
    X, Y, features = processor.process_dataset(
        output_path=args.output,
        max_samples=args.max_samples,
        extract_features=not args.no_features
    )
    
    logger.info(f"\nProcessamento concluído!")
    logger.info(f"Dados salvos em: {args.output}")
    
    # Plotar exemplo (opcional)
    if len(X) > 0:
        plot_ecg_example(X[0], processor.LEAD_NAMES, args.sampling_rate)


def plot_ecg_example(ecg_12lead: np.ndarray, lead_names: List[str], sampling_rate: int):
    """Plota um exemplo de ECG processado"""
    fig, axes = plt.subplots(12, 1, figsize=(12, 16), sharex=True)
    
    time_axis = np.arange(ecg_12lead.shape[1]) / sampling_rate
    
    for i, (ax, lead_name) in enumerate(zip(axes, lead_names)):
        ax.plot(time_axis, ecg_12lead[i, :], 'b-', linewidth=0.5)
        ax.set_ylabel(lead_name)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-3, 3)
    
    axes[-1].set_xlabel('Tempo (s)')
    fig.suptitle('ECG 12 Derivações Processado', fontsize=16)
    plt.tight_layout()
    plt.savefig('ecg_example.png', dpi=150)
    logger.info("Exemplo de ECG salvo em: ecg_example.png")


if __name__ == "__main__":
    main()
