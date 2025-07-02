#!/usr/bin/env python3
"""
Script para aplicar todas as correções nos arquivos.
"""

import os
import shutil
from datetime import datetime

def backup_file(filepath):
    """Cria backup de um arquivo."""
    if os.path.exists(filepath):
        backup_path = f"{filepath}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(filepath, backup_path)
        print(f"[BACKUP] {filepath} -> {backup_path}")

def main():
    print("="*60)
    print("APLICANDO TODAS AS CORREÇÕES")
    print("="*60)
    
    # Lista de arquivos para backup
    files_to_backup = [
        "app/models/ecg.py",
        "app/schemas/ecg.py",
        "app/services/ecg_service.py"
    ]
    
    # Criar backups
    print("\n[1/3] Criando backups...")
    for filepath in files_to_backup:
        backup_file(filepath)
    
    print("\n[2/3] Arquivos prontos para atualização!")
    print("\nAgora você deve:")
    print("1. Copiar o conteúdo de cada arquivo fornecido acima")
    print("2. Substituir o conteúdo dos arquivos correspondentes")
    print("3. Salvar todos os arquivos")
    
    print("\n[3/3] Após atualizar os arquivos, execute:")
    print("pytest tests/test_ecg_service_critical_coverage.py -v")
    print("\nPara verificar cobertura completa:")
    print("pytest --cov=app --cov-report=term-missing --cov-report=html")

if __name__ == "__main__":
    main()
