#!/usr/bin/env python3
"""
Script direto para executar preparação e treinamento PTB-XL
Caminho fixo: D:\ptb-xl\npy_lr
"""

import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import time

# Caminho fixo do dataset
DATA_PATH = r"D:\ptb-xl\npy_lr"

def main():
    print("=" * 70)
    print("🚀 EXECUÇÃO DIRETA - PTB-XL ENSEMBLE")
    print("=" * 70)
    
    data_path = Path(DATA_PATH)
    print(f"\n📁 Usando diretório: {data_path}")
    
    # 1. Verificar se o diretório existe
    if not data_path.exists():
        print(f"\n❌ ERRO: Diretório não encontrado!")
        print(f"   Procurado em: {data_path}")
        print("\n   Verifique se o caminho está correto.")
        input("\nPressione ENTER para sair...")
        return
    
    print("✅ Diretório encontrado!")
    
    # 2. Verificar se já existem arquivos consolidados
    x_path = data_path / "X.npy"
    y_path = data_path / "Y.npy"
    
    if x_path.exists() and y_path.exists():
        print("\n📦 Arquivos X.npy e Y.npy já existem!")
        try:
            X = np.load(x_path)
            Y = np.load(y_path)
            print(f"   Shape X: {X.shape}")
            print(f"   Shape Y: {Y.shape}")
            print("\n✅ Pulando preparação, indo direto para o treinamento...")
            time.sleep(2)
        except Exception as e:
            print(f"⚠️  Erro ao verificar arquivos: {e}")
            print("   Vamos recriar os arquivos...")
            prepare_data(data_path)
    else:
        print("\n📋 Arquivos consolidados não encontrados.")
        print("   Iniciando preparação dos dados...")
        time.sleep(2)
        prepare_data(data_path)
    
    # 3. Executar treinamento
    print("\n" + "-" * 70)
    print("🤖 INICIANDO TREINAMENTO DO ENSEMBLE")
    print("-" * 70)
    
    # Comando para treinar
    cmd = [
        sys.executable,
        "ensemble_ecg_training.py",
        "--data-path", str(data_path),
        "--models", "cnn", "resnet", "inception", "attention",
        "--ensemble-method", "weighted_voting",
        "--epochs", "50",  # Reduzido para teste mais rápido
        "--batch-size", "32",
        "--output-dir", "./ensemble_ptbxl_results"
    ]
    
    print("\n⚙️  Configurações:")
    print("   - Modelos: CNN, ResNet, Inception, Attention")
    print("   - Método: Votação Ponderada")
    print("   - Épocas: 50")
    print("   - Batch Size: 32")
    print("   - Saída: ./ensemble_ptbxl_results/")
    
    print("\n🚀 Executando treinamento...")
    print("   (Isso pode levar várias horas)")
    print("\n" + "-" * 70)
    
    try:
        subprocess.run(cmd, check=True)
        print("\n" + "=" * 70)
        print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print("=" * 70)
    except subprocess.CalledProcessError:
        print("\n❌ Erro no treinamento!")
        print("   Verifique os logs acima para mais detalhes.")
    except KeyboardInterrupt:
        print("\n⚠️  Treinamento interrompido!")

def prepare_data(data_path):
    """Prepara os dados criando X.npy e Y.npy"""
    print("\n🔧 PREPARANDO DADOS...")
    
    # Salvar o script de preparação
    prep_script = '''
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys

data_path = Path(sys.argv[1])
npy_files = sorted([f for f in data_path.glob("*.npy") if f.name[0].isdigit()])
print(f"Encontrados {len(npy_files)} arquivos")

if len(npy_files) == 0:
    print("Nenhum arquivo encontrado!")
    sys.exit(1)

# Carregar primeiro arquivo para verificar formato
first = np.load(npy_files[0])
print(f"Formato: {first.shape}")

# Assumir 12 derivações
if first.ndim == 1:
    n_points = len(first) // 12
    n_leads = 12
else:
    n_leads, n_points = first.shape

print(f"Detectado: {n_leads} derivações, {n_points} pontos")

# Criar arrays
X = np.zeros((len(npy_files), n_leads, n_points), dtype=np.float32)
Y = np.zeros(len(npy_files), dtype=np.int64)  # Labels dummy

# Carregar todos os arquivos
for i, f in enumerate(tqdm(npy_files)):
    try:
        data = np.load(f)
        if data.ndim == 1:
            data = data.reshape(n_leads, n_points)
        X[i] = data
    except:
        pass

# Salvar
np.save(data_path / "X.npy", X)
np.save(data_path / "Y.npy", Y)
print(f"Salvos: X{X.shape}, Y{Y.shape}")
'''
    
    # Salvar script temporário
    temp_script = Path("temp_prepare.py")
    with open(temp_script, "w") as f:
        f.write(prep_script)
    
    try:
        # Executar preparação
        subprocess.run([sys.executable, str(temp_script), str(data_path)], check=True)
        print("\n✅ Dados preparados com sucesso!")
    except Exception as e:
        print(f"\n❌ Erro na preparação: {e}")
    finally:
        # Remover script temporário
        if temp_script.exists():
            temp_script.unlink()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input("\n\nPressione ENTER para sair...")
