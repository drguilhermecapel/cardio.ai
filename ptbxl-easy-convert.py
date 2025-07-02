#!/usr/bin/env python3
"""
Script Simplificado para Conversão PTB-XL
=========================================

Executa a conversão de WFDB para NPY com configurações otimizadas.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_wfdb():
    """Verifica e instala wfdb se necessário"""
    try:
        import wfdb
        print("✓ wfdb instalado")
        return True
    except ImportError:
        print("📦 Instalando wfdb...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "wfdb"])
            print("✓ wfdb instalado com sucesso")
            return True
        except:
            print("❌ Erro ao instalar wfdb")
            print("   Tente manualmente: pip install wfdb")
            return False


def main():
    print("="*70)
    print("   🏥 CONVERSÃO FÁCIL PTB-XL")
    print("="*70)
    print()
    
    # Caminho padrão
    base_path = r"C:\Users\User\OneDrive\Documentos\GitHub\cardio.ai.pro2\ptbxl_processing\ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
    
    # Verificar se o caminho existe
    if not Path(base_path).exists():
        print(f"❌ Caminho não encontrado: {base_path}")
        print("\nPor favor, verifique o caminho do dataset PTB-XL")
        input("\nPressione ENTER para sair...")
        return
    
    print(f"📁 Dataset encontrado: {base_path}")
    
    # Verificar wfdb
    if not check_wfdb():
        input("\nPressione ENTER para sair...")
        return
    
    # Menu de opções
    print("\n🔧 OPÇÕES DE CONVERSÃO:")
    print("1. Converter 100 Hz (recomendado para início)")
    print("2. Converter 500 Hz (mais detalhado)")
    print("3. Converter ambos")
    print("4. Sair")
    
    escolha = input("\nEscolha uma opção (1-4): ").strip()
    
    if escolha == '4':
        print("Saindo...")
        return
    
    # Determinar quais taxas converter
    if escolha == '1':
        rates = [100]
    elif escolha == '2':
        rates = [500]
    elif escolha == '3':
        rates = [100, 500]
    else:
        print("Opção inválida")
        return
    
    # Confirmar
    print(f"\n📋 RESUMO:")
    print(f"   Dataset: {Path(base_path).name}")
    print(f"   Taxas: {rates} Hz")
    print(f"   Pré-processamento: Sim (filtros + normalização)")
    
    confirmar = input("\n🚀 Iniciar conversão? (S/n): ").strip().lower()
    if confirmar == 'n':
        print("Conversão cancelada")
        return
    
    # Executar conversões
    for rate in rates:
        print(f"\n{'='*70}")
        print(f"   CONVERTENDO {rate} Hz")
        print(f"{'='*70}\n")
        
        # Comando para executar o conversor principal
        cmd = [
            sys.executable,
            "ptbxl_wfdb_to_npy_converter.py",  # Nome do arquivo do conversor principal
            "--base-path", base_path,
            "--sampling-rate", str(rate)
        ]
        
        try:
            # Verificar se o script principal existe
            if not Path("ptbxl_wfdb_to_npy_converter.py").exists():
                # Criar o script principal
                print("⚠️  Criando script conversor principal...")
                create_main_converter()
            
            # Executar conversão
            subprocess.run(cmd, check=True)
            
        except subprocess.CalledProcessError:
            print(f"\n❌ Erro ao converter {rate} Hz")
            continuar = input("Continuar com próxima taxa? (S/n): ").strip().lower()
            if continuar == 'n':
                break
        except FileNotFoundError:
            print("\n❌ Script conversor não encontrado!")
            print("   Certifique-se de que ptbxl_wfdb_to_npy_converter.py está no mesmo diretório")
            break
    
    print("\n" + "="*70)
    print("   PROCESSO FINALIZADO")
    print("="*70)
    
    # Verificar resultados
    output_base = Path(base_path) / 'processed_npy'
    if output_base.exists():
        print("\n📊 RESULTADOS:")
        for rate_dir in output_base.iterdir():
            if rate_dir.is_dir():
                print(f"\n📁 {rate_dir.name}:")
                npy_files = list(rate_dir.glob("*.npy"))
                for f in sorted(npy_files):
                    size_mb = f.stat().st_size / (1024**2)
                    print(f"   - {f.name}: {size_mb:.1f} MB")
    
    print("\n✅ Conversão concluída!")
    print("\nPróximos passos:")
    print("1. Os dados convertidos estão em: processed_npy/")
    print("2. Use X.npy (sinais) e Y.npy (labels) para treinamento")
    print("3. Consulte metadata.json para informações detalhadas")
    
    input("\nPressione ENTER para sair...")


def create_main_converter():
    """Cria o script conversor principal se não existir"""
    # Aqui você colocaria o código do conversor principal
    # Por simplicidade, vamos apenas avisar o usuário
    print("Por favor, certifique-se de que o arquivo")
    print("'ptbxl_wfdb_to_npy_converter.py' está no diretório atual")
    print("\nVocê pode copiar o código do artifact anterior")
    input("\nPressione ENTER após copiar o arquivo...")


if __name__ == "__main__":
    main()
