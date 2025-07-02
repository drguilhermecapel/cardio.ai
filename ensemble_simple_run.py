#!/usr/bin/env python3
"""
Script simplificado para treinar ensemble ECG com PTB-XL
Uso: python run_ensemble_simple.py
"""

import os
import sys
import subprocess

def check_dependencies():
    """Verifica e instala dependências necessárias"""
    required = ['torch', 'numpy', 'pandas', 'scikit-learn', 'tqdm']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Instalando dependências faltantes: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)

def main():
    print("=" * 60)
    print("   TREINAMENTO ENSEMBLE ECG - MODO FÁCIL")
    print("=" * 60)
    print()
    
    # Verificar dependências
    print("Verificando dependências...")
    check_dependencies()
    print("✓ Dependências OK")
    print()
    
    # Configurações padrão - AJUSTE AQUI SE NECESSÁRIO
    DATA_PATH = r"D:\ptb-xl\npy_lr"
    
    # Verificar se o caminho existe
    if not os.path.exists(DATA_PATH):
        print(f"ERRO: Caminho não encontrado: {DATA_PATH}")
        print("Por favor, ajuste o DATA_PATH neste script.")
        input("Pressione ENTER para sair...")
        return
    
    print("CONFIGURAÇÕES:")
    print(f"📁 Dados: {DATA_PATH}")
    print("🤖 Modelos: CNN + ResNet + Inception + Attention")
    print("🎯 Método: Votação Ponderada (melhor performance)")
    print("🔄 Épocas: 100")
    print("📦 Batch: 32")
    print()
    
    resposta = input("Iniciar treinamento? (S/N): ").upper()
    if resposta != 'S':
        print("Treinamento cancelado.")
        return
    
    print()
    print("Iniciando treinamento do ensemble...")
    print("Isso pode levar várias horas dependendo do hardware.")
    print()
    
    # Comando para executar
    cmd = [
        sys.executable,
        "ensemble_ecg_training.py",
        "--data-path", DATA_PATH,
        "--models", "cnn", "resnet", "inception", "attention",
        "--ensemble-method", "weighted_voting",
        "--epochs", "100",
        "--batch-size", "32",
        "--output-dir", "./ensemble_ptbxl_results"
    ]
    
    try:
        # Executar o treinamento
        result = subprocess.run(cmd, check=True)
        
        print()
        print("=" * 60)
        print("✅ TREINAMENTO CONCLUÍDO COM SUCESSO!")
        print("=" * 60)
        print()
        print("📊 Resultados salvos em: ./ensemble_ptbxl_results/")
        print("   - ensemble_best.pth: Modelo ensemble final")
        print("   - results.json: Métricas de performance")
        print("   - model_*_best.pth: Modelos individuais")
        print()
        print("Para usar o modelo treinado:")
        print("   1. Carregue ensemble_best.pth")
        print("   2. Use o mesmo pré-processamento dos dados")
        print("   3. Faça predições com model(data)")
        
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print("❌ ERRO NO TREINAMENTO!")
        print("=" * 60)
        print()
        print("Possíveis soluções:")
        print("1. Verifique se os dados estão no formato correto (X.npy, Y.npy)")
        print("2. Reduza o batch size se houver erro de memória")
        print("3. Verifique se tem GPU disponível (nvidia-smi)")
        print("4. Certifique-se que ensemble_ecg_training.py está no mesmo diretório")
        
    except FileNotFoundError:
        print()
        print("ERRO: Arquivo ensemble_ecg_training.py não encontrado!")
        print("Certifique-se de que todos os arquivos estão no mesmo diretório.")
    
    print()
    input("Pressione ENTER para sair...")

if __name__ == "__main__":
    main()
