#!/usr/bin/env python3
"""
Script Completo para Download e Conversão do PTB-XL para formato .npy
Desenvolvido para Google Colab - Execução automática em ~15-30 minutos
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import ast
from datetime import datetime

print("="*60)
print("🏥 PTB-XL ECG Dataset - Conversor Automático para .npy")
print("="*60)
print(f"Iniciado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============== CONFIGURAÇÕES ==============
# Escolha a frequência de amostragem: 100 ou 500 Hz
SAMPLING_RATE = 100  # 100Hz = arquivos menores, 500Hz = maior resolução

# Escolha quantos ECGs processar (None = todos os 21799)
MAX_ECGS = None  # Use um número menor (ex: 1000) para teste rápido

# Dividir em batches para economizar memória
BATCH_SIZE = 500  # Processar 500 ECGs por vez
# ==========================================

# Instalar dependências
print("📦 Instalando dependências...")
os.system('pip install -q wfdb')
import wfdb

# Criar diretório de trabalho
WORK_DIR = "ptbxl_processing"
os.makedirs(WORK_DIR, exist_ok=True)
os.chdir(WORK_DIR)

# Download do dataset
print("\n📥 Baixando dataset PTB-XL (~1.7GB)...")
print("   Isso pode levar alguns minutos dependendo da conexão...")

if not os.path.exists("ptb-xl-1.0.3.zip"):
    os.system('wget -q --show-progress https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip -O ptb-xl-1.0.3.zip')
    print("✅ Download concluído!")
else:
    print("✅ Arquivo já existe, pulando download...")

# Extrair arquivos
print("\n📦 Extraindo arquivos...")
if not os.path.exists("ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"):
    os.system('unzip -q ptb-xl-1.0.3.zip')
    print("✅ Extração concluída!")
else:
    print("✅ Arquivos já extraídos...")

# Definir caminho base
BASE_PATH = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/"

# Carregar metadados
print("\n📊 Carregando metadados...")
metadata = pd.read_csv(os.path.join(BASE_PATH, 'ptbxl_database.csv'), index_col='ecg_id')
metadata.scp_codes = metadata.scp_codes.apply(lambda x: ast.literal_eval(x))

# Carregar informações de diagnóstico
scp_statements = pd.read_csv(os.path.join(BASE_PATH, 'scp_statements.csv'), index_col=0)
scp_statements = scp_statements[scp_statements.diagnostic == 1]

# Função para agregar diagnósticos
def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in scp_statements.index:
            tmp.append(scp_statements.loc[key].diagnostic_class)
    return list(set(tmp))

# Adicionar superclasses diagnósticas
metadata['diagnostic_superclass'] = metadata.scp_codes.apply(aggregate_diagnostic)

# Limitar número de ECGs se especificado
if MAX_ECGS:
    metadata = metadata[:MAX_ECGS]
    print(f"⚠️  Processando apenas {MAX_ECGS} ECGs (modo teste)")

total_ecgs = len(metadata)
print(f"\n📈 Total de ECGs a processar: {total_ecgs}")
print(f"📏 Frequência de amostragem: {SAMPLING_RATE}Hz")
print(f"⏱️  Duração de cada ECG: 10 segundos")

# Função para carregar ECGs em batches
def load_ecg_batch(metadata_batch, sampling_rate, base_path):
    """Carrega um batch de ECGs"""
    signals = []
    valid_indices = []
    
    if sampling_rate == 100:
        filenames = metadata_batch.filename_lr.values
    else:
        filenames = metadata_batch.filename_hr.values
    
    for idx, filename in enumerate(filenames):
        try:
            # Remove extensão .hea se presente
            if filename.endswith('.hea'):
                filename = filename[:-4]
            
            signal, fields = wfdb.rdsamp(os.path.join(base_path, filename))
            signals.append(signal)
            valid_indices.append(idx)
            
        except Exception as e:
            print(f"⚠️  Erro ao carregar {filename}: {str(e)}")
            continue
    
    return np.array(signals), valid_indices

# Processar ECGs em batches
print(f"\n🔄 Processando ECGs em batches de {BATCH_SIZE}...")
all_signals = []
all_valid_metadata = []

start_time = time.time()
for i in range(0, total_ecgs, BATCH_SIZE):
    batch_end = min(i + BATCH_SIZE, total_ecgs)
    batch_metadata = metadata.iloc[i:batch_end]
    
    print(f"\r   Processando ECGs {i+1}-{batch_end} de {total_ecgs}...", end='')
    
    # Carregar batch
    batch_signals, valid_indices = load_ecg_batch(batch_metadata, SAMPLING_RATE, BASE_PATH)
    
    if len(batch_signals) > 0:
        all_signals.append(batch_signals)
        all_valid_metadata.append(batch_metadata.iloc[valid_indices])

# Concatenar todos os batches
print("\n\n📊 Concatenando dados...")
X = np.vstack(all_signals)
metadata_final = pd.concat(all_valid_metadata)

processing_time = time.time() - start_time
print(f"✅ Processamento concluído em {processing_time:.1f} segundos!")

# Informações sobre os dados
print(f"\n📐 Formato final dos dados:")
print(f"   - Sinais ECG: {X.shape}")
print(f"     • {X.shape[0]} ECGs válidos")
print(f"     • {X.shape[1]} pontos temporais")
print(f"     • {X.shape[2]} derivações (leads)")
print(f"   - Metadados: {metadata_final.shape}")

# Extrair labels para classificação
print("\n🏷️  Preparando labels...")

# Labels diagnósticas (multi-label)
diagnostic_labels = metadata_final['diagnostic_superclass'].values

# Labels binárias para arritmia (exemplo)
has_arrhythmia = metadata_final['diagnostic_superclass'].apply(
    lambda x: 1 if 'CD' in x or 'HYP' in x else 0
).values

# Informações dos pacientes
patient_info = metadata_final[['patient_id', 'age', 'sex', 'height', 'weight']].copy()

# Salvar arquivos .npy
print("\n💾 Salvando arquivos...")
os.chdir('..')  # Voltar ao diretório principal

# Nome dos arquivos baseado na configuração
suffix = f"{SAMPLING_RATE}hz"
if MAX_ECGS:
    suffix += f"_{MAX_ECGS}samples"

# Salvar sinais ECG
np.save(f'ptbxl_signals_{suffix}.npy', X)
print(f"   ✅ ptbxl_signals_{suffix}.npy")

# Salvar metadados como array
np.save(f'ptbxl_metadata_{suffix}.npy', metadata_final.to_numpy())
print(f"   ✅ ptbxl_metadata_{suffix}.npy")

# Salvar labels
np.save(f'ptbxl_diagnostic_labels_{suffix}.npy', diagnostic_labels)
print(f"   ✅ ptbxl_diagnostic_labels_{suffix}.npy")

np.save(f'ptbxl_arrhythmia_labels_{suffix}.npy', has_arrhythmia)
print(f"   ✅ ptbxl_arrhythmia_labels_{suffix}.npy")

# Salvar informações dos pacientes
patient_info.to_csv(f'ptbxl_patient_info_{suffix}.csv', index=False)
print(f"   ✅ ptbxl_patient_info_{suffix}.csv")

# Criar arquivo de informações
info_text = f"""
PTB-XL Dataset - Informações de Conversão
=========================================
Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Frequência: {SAMPLING_RATE}Hz
Total ECGs: {X.shape[0]}
Tempo de processamento: {processing_time:.1f}s

Arquivos gerados:
- ptbxl_signals_{suffix}.npy: Sinais ECG {X.shape}
- ptbxl_metadata_{suffix}.npy: Metadados completos
- ptbxl_diagnostic_labels_{suffix}.npy: Labels diagnósticas
- ptbxl_arrhythmia_labels_{suffix}.npy: Labels binárias (arritmia)
- ptbxl_patient_info_{suffix}.csv: Informações dos pacientes

Derivações (ordem):
0: I, 1: II, 2: III, 3: aVR, 4: aVL, 5: aVF
6: V1, 7: V2, 8: V3, 9: V4, 10: V5, 11: V6

Classes diagnósticas principais:
NORM: Normal ECG
MI: Myocardial Infarction  
STTC: ST/T Change
CD: Conduction Disturbance
HYP: Hypertrophy
"""

with open(f'ptbxl_info_{suffix}.txt', 'w') as f:
    f.write(info_text)

print(f"\n📄 Arquivo de informações salvo: ptbxl_info_{suffix}.txt")

# Exemplo de como carregar os dados
print("\n📖 Exemplo de uso:")
print("="*50)
print("import numpy as np")
print(f"X = np.load('ptbxl_signals_{suffix}.npy')")
print(f"y = np.load('ptbxl_diagnostic_labels_{suffix}.npy', allow_pickle=True)")
print("print(f'Loaded {X.shape[0]} ECGs with shape {X.shape}')")
print("="*50)

# Download automático no Colab
try:
    from google.colab import files
    print("\n💾 Baixando arquivos para seu computador...")
    print("   (Uma janela de download deve aparecer)")
    
    # Baixar apenas os arquivos principais
    files.download(f'ptbxl_signals_{suffix}.npy')
    files.download(f'ptbxl_diagnostic_labels_{suffix}.npy')
    files.download(f'ptbxl_info_{suffix}.txt')
    
    print("\n✅ Download iniciado! Verifique sua pasta de downloads.")
except:
    print("\n⚠️  Não está rodando no Colab - arquivos salvos localmente")

# Limpar arquivos temporários (opcional)
print("\n🧹 Limpando arquivos temporários...")
if os.path.exists(WORK_DIR):
    os.system(f'rm -rf {WORK_DIR}')
    
print(f"\n✨ Processo completo! Finalizado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)