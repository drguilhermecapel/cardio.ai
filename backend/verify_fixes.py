#!/usr/bin/env python3
"""
Script para verificar se todas as correções foram aplicadas corretamente.
"""

import os
from pathlib import Path
import ast
import importlib.util

# Cores
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def check_file_exists(file_path, description):
    """Verifica se um arquivo existe."""
    if Path(file_path).exists():
        print(f"{GREEN}✓{RESET} {description}")
        return True
    else:
        print(f"{RED}✗{RESET} {description}")
        return False

def check_content_in_file(file_path, content, description):
    """Verifica se um conteúdo existe em um arquivo."""
    try:
        with open(file_path, 'r') as f:
            file_content = f.read()
        if content in file_content:
            print(f"{GREEN}✓{RESET} {description}")
            return True
        else:
            print(f"{RED}✗{RESET} {description}")
            return False
    except Exception as e:
        print(f"{RED}✗{RESET} {description} - Erro: {e}")
        return False

def check_python_syntax(file_path, description):
    """Verifica se a sintaxe Python está correta."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        ast.parse(content)
        print(f"{GREEN}✓{RESET} {description}")
        return True
    except SyntaxError as e:
        print(f"{RED}✗{RESET} {description} - Erro de sintaxe: {e}")
        return False
    except Exception as e:
        print(f"{RED}✗{RESET} {description} - Erro: {e}")
        return False

def main():
    """Executa todas as verificações."""
    print(f"\n{YELLOW}=== Verificação de Correções ==={RESET}\n")
    
    checks_passed = 0
    total_checks = 0
    
    # Base paths
    backend_dir = Path("backend")
    app_dir = backend_dir / "app"
    
    # 1. Verificar arquivo de interfaces
    total_checks += 1
    if check_file_exists(
        app_dir / "services" / "interfaces.py",
        "Arquivo de interfaces criado"
    ):
        checks_passed += 1
    
    # 2. Verificar constantes
    constants_file = app_dir / "core" / "constants.py"
    
    total_checks += 1
    if check_content_in_file(
        constants_file,
        'ROUTINE = "routine"',
        "ClinicalUrgency.ROUTINE adicionado"
    ):
        checks_passed += 1
    
    total_checks += 1
    if check_content_in_file(
        constants_file,
        'ECG_ANALYSIS_COMPLETE = "ecg_analysis_complete"',
        "NotificationType.ECG_ANALYSIS_COMPLETE adicionado"
    ):
        checks_passed += 1
    
    total_checks += 1
    if check_content_in_file(
        constants_file,
        'CSV = "csv"',
        "FileType.CSV adicionado"
    ):
        checks_passed += 1
    
    # 3. Verificar construtores
    ecg_service_file = app_dir / "services" / "ecg_service.py"
    
    total_checks += 1
    if check_content_in_file(
        ecg_service_file,
        "**kwargs",
        "ECGAnalysisService aceita kwargs"
    ):
        checks_passed += 1
    
    # 4. Verificar métodos implementados
    total_checks += 1
    if check_file_exists(
        app_dir / "utils" / "feature_extractor.py",
        "FeatureExtractor criado"
    ):
        checks_passed += 1
        
        total_checks += 1
        if check_content_in_file(
            app_dir / "utils" / "feature_extractor.py",
            "_extract_hrv_features",
            "Método _extract_hrv_features implementado"
        ):
            checks_passed += 1
    
    # 5. Verificar utilidades
    total_checks += 1
    if check_file_exists(
        app_dir / "utils" / "memory_monitor.py",
        "MemoryMonitor criado"
    ):
        checks_passed += 1
        
        total_checks += 1
        if check_content_in_file(
            app_dir / "utils" / "memory_monitor.py",
            "process_memory_mb",
            "Chave process_memory_mb adicionada"
        ):
            checks_passed += 1
    
    # 6. Verificar sintaxe Python dos arquivos principais
    print(f"\n{YELLOW}--- Verificação de Sintaxe ---{RESET}\n")
    
    files_to_check = [
        (ecg_service_file, "ECGAnalysisService sintaxe válida"),
        (constants_file, "Constants sintaxe válida"),
        (app_dir / "services" / "interfaces.py", "Interfaces sintaxe válida"),
    ]
    
    for file_path, description in files_to_check:
        if file_path.exists():
            total_checks += 1
            if check_python_syntax(file_path, description):
                checks_passed += 1
    
    # Resumo
    print(f"\n{YELLOW}=== Resumo ==={RESET}")
    print(f"Total de verificações: {total_checks}")
    print(f"Verificações bem-sucedidas: {checks_passed}")
    print(f"Taxa de sucesso: {(checks_passed/total_checks)*100:.1f}%")
    
    if checks_passed == total_checks:
        print(f"\n{GREEN}✓ Todas as correções foram aplicadas com sucesso!{RESET}")
        return 0
    else:
        print(f"\n{RED}✗ Algumas correções falharam. Execute o script fix_all_tests_coverage.py novamente.{RESET}")
        return 1

if __name__ == "__main__":
    exit(main())
