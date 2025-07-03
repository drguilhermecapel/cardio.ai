#!/usr/bin/env python3
"""
Script para instalar todas as dependências do projeto Cardio.AI Pro.
"""

import subprocess
import sys
import os

def main():
    print("="*60)
    print("INSTALAÇÃO DE DEPENDÊNCIAS - CARDIO.AI PRO")
    print("="*60)
    
    # Verificar se está no diretório backend
    if not os.path.exists("app"):
        print("[ERRO] Execute este script do diretório 'backend/'")
        return
    
    # Atualizar pip
    print("\n[1/3] Atualizando pip...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # Instalar dependências principais
    print("\n[2/3] Instalando dependências principais...")
    if os.path.exists("requirements.txt"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("[OK] Dependências principais instaladas")
    else:
        print("[AVISO] requirements.txt não encontrado - criando...")
        # Criar requirements.txt com o conteúdo fornecido acima
        
    # Instalar dependências de desenvolvimento (se existir)
    print("\n[3/3] Verificando dependências de desenvolvimento...")
    if os.path.exists("requirements-dev.txt"):
        response = input("Instalar dependências de desenvolvimento? (s/n): ")
        if response.lower() == 's':
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"])
            print("[OK] Dependências de desenvolvimento instaladas")
    
    print("\n[CONCLUÍDO] Todas as dependências foram instaladas!")
    print("\nPróximos passos:")
    print("1. Configure as variáveis de ambiente no arquivo .env")
    print("2. Execute as migrações: alembic upgrade head")
    print("3. Inicie o servidor: uvicorn app.main:app --reload")

if __name__ == "__main__":
    main()
