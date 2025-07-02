#!/usr/bin/env python3
"""
Script PTB-XL para Windows - Download e Conversão Automática para .npy
Compatível com Windows/PowerShell
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import ast
import urllib.request
import zipfile
from datetime import datetime
import shutil

print("="*60)
print("🏥 PTB-XL ECG Dataset - Conversor Automático para .npy")
print("🖥️  Versão Windows")
print("="*60)
print(f"Iniciado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============== CONFIGURAÇÕES ==============
# Escolha a frequência de amostragem: 100 ou 500 Hz
SAMPLING_RATE = 100  # 100Hz = arquivos menores, 500Hz = maior resolução

# Escolha quantos ECGs processar (None = todos os 21799)
MAX_ECGS = 1000  # Comece com 1000 para teste

# Dividir em batches para economizar memória
BATCH_SIZE = 100  # Processar 100 ECGs por vez no Windows
# ==========================================

# Instalar dependências
print("📦 Verificando dependências...")
try:
    import wfdb
    print("✅ wfdb já instalado")
except ImportError:
    print("📦 Instalando wfdb...")
    os.system(f"{sys.executable} -m pip install wfdb")
    import wfdb

# Criar diretório de trabalho
WORK_DIR = "ptbxl_processing"
if not os.path.exists(WORK_DIR):
    os.makedirs(WORK_DIR)
os.chdir(WORK_DIR)

# Download do dataset usando urllib (funciona no Windows)
print("\n📥 Baixando dataset PTB-XL (~1.7GB)...")
print("   Isso pode levar alguns minutos dependendo da conexão...")

zip_filename = "ptb-xl-1.0.3.zip"
dataset_url = "https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip"

if not os.path.exists(zip_filename):
    def download_with_progress(url, filename):
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            mb_downloaded = downloaded / 1024 / 1024
            mb_total = total_size / 1024 / 1024
            print(f"   Baixando: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='\r')
        
        urllib.request.urlretrieve(url, filename, reporthook=download_progress)
        print()  # Nova linha após o download
    
    try:
        download_with_progress(dataset_url, zip_filename)
        print("✅ Download concluído!")
    except Exception as e:
        print(f"❌ Erro no download: {e}")
        print("\nAlternativa: Baixe manualmente de:")
        print(f"   {dataset_url}")
        print(f"   E coloque o arquivo na pasta: {os.getcwd()}")
        sys.exit(1)
else:
    print("✅ Arquivo já existe, pulando download...")

# Extrair arquivos usando zipfile (funciona no Windows)
print("\n📦 Extraindo arquivos...")
extracted_folder = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"

if not os.path.exists(extracted_folder):
    try:
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            # Extrair com progresso
            files = zip_ref.namelist()
            total_files = len(files)
            print(f"   Total de arquivos: {total_files}")
            
            for i, file in enumerate(files):
                if i % 100 == 0:
                    print(f"   Extraindo: {i}/{total_files} arquivos ({i/total_files*100:.1f}%)", end='\r')
                zip_ref.extract(file)
            
        print(f"\n✅ Extração concluída!")
    except Exception as e:
        print(f"❌ Erro na extração: {e}")
        sys.exit(1)
else:
    print("✅ Arquivos já extraídos...")

# Definir caminho base
BASE_PATH = extracted_folder + "/"

# Verificar se os arquivos existem
if not os.path.exists(os.path.join(BASE_PATH, 'ptbxl_database.csv')):
    print(f"❌ Erro: Arquivo ptbxl_database.csv não encontrado em {BASE_PATH}")
    print("   Verifique se o download e extração foram bem sucedidos.")
    sys.exit(1)

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
            
            # Corrigir caminho para Windows
            full_path = os.path.join(base_path, filename).replace('\\', '/')
            signal, fields = wfdb.rdsamp(full_path)
            signals.append(signal)
            valid_indices.append(idx)
            
        except Exception as e:
            print(f"\n⚠️  Erro ao carregar {filename}: {str(e)}")
            continue
    
    return np.array(signals) if signals else np.array([]), valid_indices

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
if all_signals:
    X = np.vstack(all_signals)
    metadata_final = pd.concat(all_valid_metadata)
else:
    print("❌ Nenhum ECG foi carregado com sucesso!")
    sys.exit(1)

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
print(f"   ✅ ptbxl_signals_{suffix}.npy ({X.nbytes / 1024 / 1024:.1f} MB)")

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

# Estatísticas finais
print(f"\n📊 Estatísticas do dataset:")
print(f"   - Total de pacientes únicos: {metadata_final['patient_id'].nunique()}")
print(f"   - Idade média: {metadata_final['age'].mean():.1f} anos")
print(f"   - Distribuição por sexo:")
print(f"     • Masculino: {(metadata_final['sex']==0).sum()} ({(metadata_final['sex']==0).sum()/len(metadata_final)*100:.1f}%)")
print(f"     • Feminino: {(metadata_final['sex']==1).sum()} ({(metadata_final['sex']==1).sum()/len(metadata_final)*100:.1f}%)")

# Limpar arquivos temporários (opcional)
print("\n🧹 Deseja limpar os arquivos temporários? (economiza ~1.7GB)")
resposta = input("   Digite 's' para sim ou 'n' para não: ").lower()
if resposta == 's':
    if os.path.exists(WORK_DIR):
        shutil.rmtree(WORK_DIR)
        print("   ✅ Arquivos temporários removidos!")
else:
    print("   ℹ️  Arquivos temporários mantidos em:", os.path.abspath(WORK_DIR))

print(f"\n✨ Processo completo! Finalizado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📁 Arquivos salvos em: {os.getcwd()}")
print("="*60)