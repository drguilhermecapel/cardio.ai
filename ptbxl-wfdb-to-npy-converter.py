#!/usr/bin/env python3
"""
Conversor Robusto PTB-XL: WFDB (.hea/.dat) para NumPy (.npy)
=============================================================

Este script converte o dataset PTB-XL completo do formato WFDB para NumPy,
incluindo processamento de sinais, extração de labels e metadados.

Características:
- Suporta ambas as versões (100Hz e 500Hz)
- Processa labels reais do dataset
- Validação de integridade dos dados
- Logging detalhado
- Recuperação de erros
- Otimizado para grandes datasets

Autor: Sistema Cardio.AI
Data: 2024
"""

import os
import sys
import time
import json
import logging
import warnings
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from datetime import datetime
import ast

import numpy as np
import pandas as pd
from tqdm import tqdm
import wfdb
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d

# Suprimir warnings não críticos
warnings.filterwarnings('ignore')

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ptbxl_conversion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class PTBXLConverter:
    """Classe principal para conversão do dataset PTB-XL"""
    
    def __init__(self, base_path: str, sampling_rate: int = 100):
        """
        Inicializa o conversor
        
        Args:
            base_path: Caminho base do dataset PTB-XL
            sampling_rate: Taxa de amostragem (100 ou 500 Hz)
        """
        self.base_path = Path(base_path)
        self.sampling_rate = sampling_rate
        self.records_path = self.base_path / f'records{sampling_rate}'
        
        # Diretórios de saída
        self.output_base = self.base_path / 'processed_npy'
        self.output_path = self.output_base / f'ptbxl_{sampling_rate}hz'
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Estatísticas
        self.stats = {
            'total_records': 0,
            'processed': 0,
            'errors': 0,
            'skipped': 0
        }
        
        # Cache para metadados
        self.metadata_cache = {}
        
        logger.info(f"Inicializando conversor PTB-XL")
        logger.info(f"Caminho base: {self.base_path}")
        logger.info(f"Taxa de amostragem: {sampling_rate} Hz")
        logger.info(f"Saída: {self.output_path}")
    
    def validate_environment(self) -> bool:
        """Valida o ambiente e estrutura do dataset"""
        logger.info("Validando ambiente...")
        
        # Verificar se o diretório base existe
        if not self.base_path.exists():
            logger.error(f"Diretório base não encontrado: {self.base_path}")
            return False
        
        # Verificar se o diretório de records existe
        if not self.records_path.exists():
            logger.error(f"Diretório de records não encontrado: {self.records_path}")
            logger.info("Estrutura esperada:")
            logger.info(f"  {self.base_path}/")
            logger.info(f"    ├── records100/")
            logger.info(f"    ├── records500/")
            logger.info(f"    ├── ptbxl_database.csv")
            logger.info(f"    └── scp_statements.csv")
            return False
        
        # Verificar arquivos essenciais
        essential_files = [
            self.base_path / 'ptbxl_database.csv',
            self.base_path / 'scp_statements.csv'
        ]
        
        for file in essential_files:
            if not file.exists():
                logger.error(f"Arquivo essencial não encontrado: {file}")
                return False
        
        # Contar registros disponíveis
        hea_files = list(self.records_path.rglob("*.hea"))
        self.stats['total_records'] = len(hea_files)
        
        logger.info(f"✓ Ambiente validado")
        logger.info(f"✓ Encontrados {self.stats['total_records']} registros")
        
        return True
    
    def load_metadata(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Carrega os metadados do dataset"""
        logger.info("Carregando metadados...")
        
        # Carregar database principal
        db_path = self.base_path / 'ptbxl_database.csv'
        df_db = pd.read_csv(db_path)
        logger.info(f"✓ Database carregado: {len(df_db)} registros")
        
        # Carregar códigos SCP
        scp_path = self.base_path / 'scp_statements.csv'
        df_scp = pd.read_csv(scp_path, index_col=0)
        logger.info(f"✓ SCP codes carregados: {len(df_scp)} códigos")
        
        # Processar códigos SCP no dataframe principal
        df_db['scp_codes_dict'] = df_db['scp_codes'].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) else {}
        )
        
        return df_db, df_scp
    
    def read_ecg_record(self, record_path: str) -> Optional[np.ndarray]:
        """
        Lê um registro ECG no formato WFDB
        
        Args:
            record_path: Caminho para o arquivo .hea (sem extensão)
            
        Returns:
            Array numpy com shape (12, n_samples) ou None se erro
        """
        try:
            # Ler registro WFDB
            record = wfdb.rdrecord(record_path)
            
            # Extrair sinais
            signals = record.p_signal  # Shape: (n_samples, n_leads)
            
            # Transpor para formato (n_leads, n_samples)
            signals = signals.T
            
            # Validar número de derivações
            if signals.shape[0] != 12:
                logger.warning(f"Número incorreto de derivações: {signals.shape[0]}")
                return None
            
            # Validar valores
            if np.any(np.isnan(signals)) or np.any(np.isinf(signals)):
                logger.warning(f"Valores inválidos encontrados no sinal")
                # Tentar corrigir
                signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalizar amplitudes (mV)
            # PTB-XL já está em mV, mas vamos garantir range razoável
            signal_range = np.ptp(signals)
            if signal_range > 10:  # Provavelmente em unidades incorretas
                signals = signals / 1000.0  # Converter para mV
            
            return signals.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Erro ao ler {record_path}: {str(e)}")
            return None
    
    def process_labels(self, df_db: pd.DataFrame, df_scp: pd.DataFrame) -> Dict[str, Any]:
        """Processa e organiza os labels do dataset"""
        logger.info("Processando labels...")
        
        # Extrair todas as condições únicas
        all_conditions = set()
        for codes in df_db['scp_codes_dict']:
            all_conditions.update(codes.keys())
        
        condition_list = sorted(list(all_conditions))
        n_conditions = len(condition_list)
        logger.info(f"✓ Identificadas {n_conditions} condições únicas")
        
        # Criar mapeamentos
        condition_to_idx = {cond: idx for idx, cond in enumerate(condition_list)}
        idx_to_condition = {idx: cond for idx, cond in enumerate(condition_list)}
        
        # Adicionar descrições das condições
        condition_descriptions = {}
        for cond in condition_list:
            if cond in df_scp.index:
                condition_descriptions[cond] = {
                    'description': df_scp.loc[cond, 'description'],
                    'diagnostic_class': df_scp.loc[cond, 'diagnostic_class'] 
                    if 'diagnostic_class' in df_scp.columns else 'Unknown'
                }
            else:
                condition_descriptions[cond] = {
                    'description': 'Unknown',
                    'diagnostic_class': 'Unknown'
                }
        
        # Estatísticas das condições
        condition_counts = defaultdict(int)
        for codes in df_db['scp_codes_dict']:
            for cond in codes.keys():
                condition_counts[cond] += 1
        
        # Top condições
        top_conditions = sorted(condition_counts.items(), 
                               key=lambda x: x[1], reverse=True)[:20]
        
        logger.info("\nTop 10 condições mais frequentes:")
        for cond, count in top_conditions[:10]:
            desc = condition_descriptions[cond]['description']
            pct = count / len(df_db) * 100
            logger.info(f"  {cond}: {count} ({pct:.1f}%) - {desc}")
        
        return {
            'condition_list': condition_list,
            'condition_to_idx': condition_to_idx,
            'idx_to_condition': idx_to_condition,
            'condition_descriptions': condition_descriptions,
            'condition_counts': dict(condition_counts),
            'n_conditions': n_conditions
        }
    
    def convert_batch(self, batch_records: List[Tuple[int, str]], 
                     df_db: pd.DataFrame, label_info: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Converte um lote de registros
        
        Returns:
            X: Array de sinais ECG
            Y_single: Labels single-label 
            Y_multi: Labels multi-label
        """
        batch_size = len(batch_records)
        n_conditions = label_info['n_conditions']
        
        # Determinar tamanho do sinal baseado na taxa de amostragem
        signal_length = 5000 if self.sampling_rate == 500 else 1000
        
        # Inicializar arrays
        X = np.zeros((batch_size, 12, signal_length), dtype=np.float32)
        Y_single = np.zeros(batch_size, dtype=np.int64)
        Y_multi = np.zeros((batch_size, n_conditions), dtype=np.float32)
        
        valid_count = 0
        
        for i, (ecg_id, record_path) in enumerate(batch_records):
            # Ler sinal ECG
            signals = self.read_ecg_record(record_path)
            
            if signals is None:
                self.stats['errors'] += 1
                continue
            
            # Ajustar comprimento do sinal se necessário
            current_length = signals.shape[1]
            if current_length != signal_length:
                # Interpolar ou truncar
                if current_length > signal_length:
                    # Truncar
                    signals = signals[:, :signal_length]
                else:
                    # Interpolar
                    x_old = np.linspace(0, 1, current_length)
                    x_new = np.linspace(0, 1, signal_length)
                    signals_new = np.zeros((12, signal_length))
                    for lead in range(12):
                        f = interp1d(x_old, signals[lead, :], kind='linear')
                        signals_new[lead, :] = f(x_new)
                    signals = signals_new
            
            X[valid_count] = signals
            
            # Processar labels
            row = df_db[df_db['ecg_id'] == ecg_id]
            if not row.empty:
                scp_codes = row.iloc[0]['scp_codes_dict']
                
                # Multi-label
                for cond, confidence in scp_codes.items():
                    if cond in label_info['condition_to_idx']:
                        idx = label_info['condition_to_idx'][cond]
                        Y_multi[valid_count, idx] = confidence / 100.0
                
                # Single-label (condição principal)
                if scp_codes:
                    main_cond = max(scp_codes.items(), key=lambda x: x[1])[0]
                    if main_cond in label_info['condition_to_idx']:
                        Y_single[valid_count] = label_info['condition_to_idx'][main_cond]
            
            valid_count += 1
            self.stats['processed'] += 1
        
        # Retornar apenas registros válidos
        return X[:valid_count], Y_single[:valid_count], Y_multi[:valid_count]
    
    def apply_preprocessing(self, signals: np.ndarray) -> np.ndarray:
        """
        Aplica pré-processamento básico aos sinais
        
        Args:
            signals: Array com shape (n_leads, n_samples)
            
        Returns:
            Sinais pré-processados
        """
        # Remover offset DC
        signals = signals - np.mean(signals, axis=1, keepdims=True)
        
        # Filtro passa-banda (0.5-40 Hz)
        nyquist = self.sampling_rate / 2
        low = 0.5 / nyquist
        high = 40.0 / nyquist
        
        if high < 1.0:  # Garantir que a frequência está no range válido
            b, a = scipy_signal.butter(3, [low, high], btype='band')
            for i in range(signals.shape[0]):
                signals[i] = scipy_signal.filtfilt(b, a, signals[i])
        
        return signals
    
    def convert_dataset(self, apply_preprocessing: bool = True):
        """Converte todo o dataset"""
        # Validar ambiente
        if not self.validate_environment():
            logger.error("Falha na validação do ambiente")
            return
        
        # Carregar metadados
        df_db, df_scp = self.load_metadata()
        
        # Processar labels
        label_info = self.process_labels(df_db, df_scp)
        
        # Listar todos os registros
        logger.info("\nListando registros para conversão...")
        all_records = []
        
        for patient_dir in sorted(self.records_path.iterdir()):
            if patient_dir.is_dir():
                for record_file in patient_dir.glob("*.hea"):
                    record_path = str(record_file.with_suffix(''))
                    ecg_id = int(record_file.stem)
                    all_records.append((ecg_id, record_path))
        
        logger.info(f"✓ Encontrados {len(all_records)} registros para processar")
        
        # Processar em lotes
        batch_size = 100
        n_batches = (len(all_records) + batch_size - 1) // batch_size
        
        all_X = []
        all_Y_single = []
        all_Y_multi = []
        all_ecg_ids = []
        
        logger.info(f"\nProcessando {n_batches} lotes...")
        
        for batch_idx in tqdm(range(n_batches), desc="Convertendo"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(all_records))
            batch_records = all_records[start_idx:end_idx]
            
            # Converter lote
            X_batch, Y_single_batch, Y_multi_batch = self.convert_batch(
                batch_records, df_db, label_info
            )
            
            # Aplicar pré-processamento se solicitado
            if apply_preprocessing and len(X_batch) > 0:
                for i in range(len(X_batch)):
                    X_batch[i] = self.apply_preprocessing(X_batch[i])
            
            # Adicionar aos resultados
            if len(X_batch) > 0:
                all_X.append(X_batch)
                all_Y_single.append(Y_single_batch)
                all_Y_multi.append(Y_multi_batch)
                all_ecg_ids.extend([r[0] for r in batch_records[:len(X_batch)]])
            
            # Salvar checkpoint a cada 1000 registros
            if self.stats['processed'] % 1000 == 0 and self.stats['processed'] > 0:
                self._save_checkpoint(all_X, all_Y_single, all_Y_multi, 
                                    all_ecg_ids, label_info)
        
        # Concatenar todos os resultados
        logger.info("\nConcatenando resultados...")
        X_final = np.concatenate(all_X, axis=0)
        Y_single_final = np.concatenate(all_Y_single, axis=0)
        Y_multi_final = np.concatenate(all_Y_multi, axis=0)
        ecg_ids_final = np.array(all_ecg_ids)
        
        # Salvar resultados finais
        self._save_final_results(X_final, Y_single_final, Y_multi_final, 
                               ecg_ids_final, label_info, df_db)
        
        # Estatísticas finais
        logger.info("\n" + "="*60)
        logger.info("CONVERSÃO CONCLUÍDA")
        logger.info("="*60)
        logger.info(f"Total de registros: {self.stats['total_records']}")
        logger.info(f"Processados com sucesso: {self.stats['processed']}")
        logger.info(f"Erros: {self.stats['errors']}")
        logger.info(f"Taxa de sucesso: {self.stats['processed']/self.stats['total_records']*100:.1f}%")
        logger.info(f"\nArquivos salvos em: {self.output_path}")
    
    def _save_checkpoint(self, all_X, all_Y_single, all_Y_multi, 
                        all_ecg_ids, label_info):
        """Salva checkpoint durante o processamento"""
        checkpoint_dir = self.output_path / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = checkpoint_dir / f'checkpoint_{timestamp}.npz'
        
        X_temp = np.concatenate(all_X, axis=0)
        Y_single_temp = np.concatenate(all_Y_single, axis=0)
        Y_multi_temp = np.concatenate(all_Y_multi, axis=0)
        
        np.savez_compressed(
            checkpoint_path,
            X=X_temp,
            Y_single=Y_single_temp,
            Y_multi=Y_multi_temp,
            ecg_ids=all_ecg_ids,
            processed_count=self.stats['processed']
        )
        
        logger.info(f"Checkpoint salvo: {checkpoint_path}")
    
    def _save_final_results(self, X, Y_single, Y_multi, ecg_ids, 
                           label_info, df_db):
        """Salva os resultados finais da conversão"""
        logger.info("\nSalvando resultados finais...")
        
        # Salvar arrays principais
        np.save(self.output_path / 'X.npy', X)
        logger.info(f"✓ X.npy salvo: {X.shape}")
        
        np.save(self.output_path / 'Y.npy', Y_single)
        logger.info(f"✓ Y.npy salvo: {Y_single.shape}")
        
        np.save(self.output_path / 'Y_multilabel.npy', Y_multi)
        logger.info(f"✓ Y_multilabel.npy salvo: {Y_multi.shape}")
        
        np.save(self.output_path / 'ecg_ids.npy', ecg_ids)
        logger.info(f"✓ ecg_ids.npy salvo: {ecg_ids.shape}")
        
        # Salvar metadados
        metadata = {
            'sampling_rate': self.sampling_rate,
            'n_samples': len(X),
            'n_leads': 12,
            'signal_length': X.shape[2],
            'n_conditions': label_info['n_conditions'],
            'condition_list': label_info['condition_list'],
            'condition_to_idx': label_info['condition_to_idx'],
            'idx_to_condition': label_info['idx_to_condition'],
            'condition_descriptions': label_info['condition_descriptions'],
            'condition_counts': label_info['condition_counts'],
            'conversion_stats': self.stats,
            'conversion_date': datetime.now().isoformat()
        }
        
        with open(self.output_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info("✓ metadata.json salvo")
        
        # Salvar informações sobre o dataset
        self._save_dataset_info(X, Y_single, Y_multi, label_info, df_db)
    
    def _save_dataset_info(self, X, Y_single, Y_multi, label_info, df_db):
        """Salva arquivo informativo sobre o dataset"""
        info_text = f"""
PTB-XL Dataset Convertido para NumPy
====================================

Data da conversão: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Taxa de amostragem: {self.sampling_rate} Hz

ESTATÍSTICAS DO DATASET
----------------------
Total de amostras: {len(X):,}
Derivações: 12
Pontos por derivação: {X.shape[2]:,}
Duração do sinal: {X.shape[2] / self.sampling_rate:.1f} segundos

LABELS
------
Condições únicas: {label_info['n_conditions']}
Tipo single-label: Condição principal por ECG
Tipo multi-label: Todas as condições com nível de confiança

TOP 10 CONDIÇÕES MAIS FREQUENTES
--------------------------------
"""
        # Adicionar top condições
        condition_counts = label_info['condition_counts']
        top_conditions = sorted(condition_counts.items(), 
                               key=lambda x: x[1], reverse=True)[:10]
        
        for i, (cond, count) in enumerate(top_conditions, 1):
            desc = label_info['condition_descriptions'][cond]['description']
            pct = count / len(X) * 100
            info_text += f"{i:2d}. {cond}: {count:,} ({pct:.1f}%) - {desc}\n"
        
        info_text += f"""
ARQUIVOS GERADOS
---------------
- X.npy: Sinais ECG ({X.shape})
- Y.npy: Labels single-label ({Y_single.shape})
- Y_multilabel.npy: Labels multi-label ({Y_multi.shape})
- ecg_ids.npy: IDs dos ECGs
- metadata.json: Metadados completos
- dataset_info.txt: Este arquivo

INFORMAÇÕES TÉCNICAS
-------------------
- Formato dos sinais: (n_samples, n_leads, n_points)
- Unidade: milivolts (mV)
- Pré-processamento: Remoção de offset DC + Filtro passa-banda 0.5-40 Hz
- Labels single: Índice da condição principal (0 a {label_info['n_conditions']-1})
- Labels multi: Matriz binária com níveis de confiança (0.0 a 1.0)

USO RECOMENDADO
--------------
1. Carregar dados:
   X = np.load('X.npy')
   Y = np.load('Y.npy')
   
2. Para multi-label:
   Y_multi = np.load('Y_multilabel.npy')
   
3. Para metadados:
   import json
   with open('metadata.json', 'r') as f:
       metadata = json.load(f)

ESTATÍSTICAS DA CONVERSÃO
------------------------
Registros processados: {self.stats['processed']:,}
Erros: {self.stats['errors']:,}
Taxa de sucesso: {self.stats['processed']/self.stats['total_records']*100:.1f}%
"""
        
        with open(self.output_path / 'dataset_info.txt', 'w', encoding='utf-8') as f:
            f.write(info_text)
        
        logger.info("✓ dataset_info.txt salvo")


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Conversor PTB-XL: WFDB para NumPy"
    )
    parser.add_argument(
        '--base-path', 
        type=str,
        default=r"C:\Users\User\OneDrive\Documentos\GitHub\cardio.ai.pro2\ptbxl_processing\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
        help='Caminho base do dataset PTB-XL'
    )
    parser.add_argument(
        '--sampling-rate',
        type=int,
        choices=[100, 500],
        default=100,
        help='Taxa de amostragem (100 ou 500 Hz)'
    )
    parser.add_argument(
        '--no-preprocessing',
        action='store_true',
        help='Pular pré-processamento dos sinais'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("   CONVERSOR PTB-XL: WFDB → NumPy")
    print("="*70)
    print()
    
    # Criar conversor
    converter = PTBXLConverter(
        base_path=args.base_path,
        sampling_rate=args.sampling_rate
    )
    
    # Executar conversão
    try:
        converter.convert_dataset(
            apply_preprocessing=not args.no_preprocessing
        )
        
        print("\n✅ Conversão concluída com sucesso!")
        print(f"📁 Arquivos salvos em: {converter.output_path}")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Conversão interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro durante a conversão: {e}")
        traceback.print_exc()
    
    input("\nPressione ENTER para sair...")


if __name__ == "__main__":
    main()
