#!/usr/bin/env python3
"""
Script para corrigir completamente o MemoryMonitor e executar testes.
"""

import os
import subprocess
import sys

def main():
    print("="*60)
    print("CORREÇÃO COMPLETA DO MEMORY MONITOR")
    print("="*60)
    
    # 1. Instalar psutil se necessário
    print("\n[1/4] Instalando dependências...")
    try:
        import psutil
        print("[OK] psutil já instalado")
    except ImportError:
        print("[INFO] Instalando psutil...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        print("[OK] psutil instalado")
    
    # 2. Verificar se o arquivo memory_monitor.py foi criado corretamente
    memory_monitor_path = os.path.join("app", "utils", "memory_monitor.py")
    if os.path.exists(memory_monitor_path):
        print(f"\n[2/4] Arquivo {memory_monitor_path} existe")
        
        # Verificar se tem a classe MemoryMonitor
        with open(memory_monitor_path, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'class MemoryMonitor' in content:
                print("[OK] Classe MemoryMonitor encontrada")
            else:
                print("[ERRO] Classe MemoryMonitor NÃO encontrada!")
                return
    else:
        print(f"[ERRO] Arquivo {memory_monitor_path} NÃO existe!")
        return
    
    # 3. Corrigir import do Celery no teste
    print("\n[3/4] Corrigindo imports do Celery...")
    test_file = os.path.join("tests", "test_ecg_tasks_complete_coverage.py")
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Corrigir o import
        if "from celery.exceptions import Retry" in content:
            content = content.replace(
                "from celery.exceptions import Retry",
                "from celery import exceptions\nRetry = exceptions.Retry"
            )
            
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print("[OK] Import do Celery corrigido")
    
    # 4. Executar testes
    print("\n[4/4] Executando testes...")
    print("-"*60)
    
    # Primeiro teste crítico
    result = subprocess.run(
        ["pytest", "tests/test_ecg_service_critical_coverage.py", "-v"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("\n[SUCESSO] Teste crítico passou!")
        
        # Executar todos os testes com cobertura
        print("\n[INFO] Executando todos os testes com cobertura...")
        subprocess.run([
            "pytest", "--cov=app", "--cov-report=term-missing", 
            "--cov-report=html", "-v"
        ])
    else:
        print("\n[ERRO] Teste crítico falhou!")
        print(result.stdout)
        print(result.stderr)

if __name__ == "__main__":
    main()
