#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script final que executa todas as correções em sequência
"""

import subprocess
import sys
from pathlib import Path

print("="*60)
print("EXECUTANDO CORREÇÕES COMPLETAS - CARDIOAI PRO")
print("="*60)

# 1. Primeiro corrigir MemoryMonitor
print("\n[1/3] Corrigindo MemoryMonitor...")
result = subprocess.run([sys.executable, "fix-memory-monitor-class.py"], capture_output=True, text=True)
if result.returncode == 0:
    print("[OK] MemoryMonitor corrigido")
else:
    print("[AVISO] Erro ao corrigir MemoryMonitor, continuando...")

# 2. Executar correções principais
print("\n[2/3] Executando correções principais...")
result = subprocess.run([sys.executable, "fix-all-tests-now.py"], capture_output=True, text=True)
print(result.stdout)
if result.stderr:
    print(result.stderr)

# 3. Executar testes críticos
print("\n[3/3] Executando testes críticos...")
print("="*60)

result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_ecg_service_critical_coverage.py", "-v", "--tb=short"],
    capture_output=True,
    text=True
)

print(result.stdout)
if result.stderr:
    print(result.stderr)

# Verificar sucesso
if "FAILED" not in result.stdout and "failed" not in result.stdout and "ERROR" not in result.stdout:
    print("\n" + "="*60)
    print("[SUCESSO] TODOS OS TESTES CRÍTICOS PASSARAM!")
    print("="*60)
    
    # Executar cobertura completa
    print("\n[BÔNUS] Executando análise de cobertura completa...")
    cov_result = subprocess.run(
        [sys.executable, "-m", "pytest", "--cov=app", "--cov-report=term-missing", "--cov-report=html", "-q"],
        capture_output=True,
        text=True
    )
    
    # Mostrar apenas o resumo
    lines = cov_result.stdout.split('\n')
    for line in lines:
        if "TOTAL" in line or "app/" in line:
            print(line)
    
    print("\n[INFO] Relatório HTML gerado em: htmlcov/index.html")
else:
    print("\n[AVISO] Ainda há problemas nos testes")
    print("Execute manualmente:")
    print("  1. python fix-memory-monitor-class.py")
    print("  2. python fix-all-tests-now.py")
    print("  3. pytest tests/test_ecg_service_critical_coverage.py -vv")

print("\n[CONCLUÍDO]")
