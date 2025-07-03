#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script final para executar todas as correções e alcançar 100% nos testes críticos
"""

import os
import sys
import subprocess
from pathlib import Path

def run_script(script_name):
    """Executa um script Python."""
    print(f"\n[INFO] Executando {script_name}...")
    try:
        result = subprocess.run([sys.executable, script_name], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"[OK] {script_name} executado com sucesso")
            return True
        else:
            print(f"[ERRO] {script_name} falhou")
            if result.stderr:
                print(f"Erro: {result.stderr}")
            return False
    except Exception as e:
        print(f"[ERRO] Não foi possível executar {script_name}: {e}")
        return False


def run_tests():
    """Executa os testes críticos."""
    print("\n[TESTE] Executando testes críticos...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_ecg_service_critical_coverage.py", "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    
    # Analisar resultado
    output = result.stdout + result.stderr
    
    if "failed" in output:
        failed_count = 0
        for line in output.split('\n'):
            if "failed" in line and "passed" in line:
                # Extrair números
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "failed":
                        try:
                            failed_count = int(parts[i-1])
                        except:
                            pass
        
        print(f"[AVISO] {failed_count} testes falharam")
        return False
    elif "passed" in output and "failed" not in output:
        print("[OK] Todos os testes passaram!")
        return True
    else:
        print("[ERRO] Não foi possível determinar o resultado dos testes")
        return False


def main():
    """Função principal."""
    print("="*60)
    print("EXECUÇÃO FINAL - CORREÇÃO COMPLETA CARDIOAI PRO")
    print("="*60)
    
    # Lista de scripts para executar em ordem
    scripts = [
        "fix-critical-tests-issues.py",
        "comprehensive-test-fix.py"
    ]
    
    # Verificar se os scripts existem
    missing_scripts = []
    for script in scripts:
        if not Path(script).exists():
            missing_scripts.append(script)
    
    if missing_scripts:
        print("\n[AVISO] Scripts faltando:")
        for script in missing_scripts:
            print(f"  - {script}")
        
        print("\n[INFO] Executando correção direta...")
        
        # Executar correções diretamente
        try:
            from comprehensive_test_fix import main as fix_main
            fix_main()
        except ImportError:
            print("[ERRO] Não foi possível importar comprehensive_test_fix")
            print("Por favor, certifique-se de que todos os scripts estão no diretório.")
            return 1
    else:
        # Executar scripts em ordem
        for script in scripts:
            if not run_script(script):
                print(f"\n[ERRO] Falha ao executar {script}")
                print("Continuando com próximo script...")
    
    # Executar testes
    print("\n" + "="*60)
    print("VERIFICANDO RESULTADOS")
    print("="*60)
    
    if run_tests():
        print("\n" + "="*60)
        print("[SUCESSO] TESTES CRÍTICOS PASSANDO 100%!")
        print("="*60)
        
        print("\n[PRÓXIMO PASSO] Execute a cobertura completa:")
        print("  pytest --cov=app --cov-report=html --cov-report=term-missing")
        
        # Tentar executar cobertura automaticamente
        print("\n[INFO] Executando cobertura completa...")
        cov_result = subprocess.run(
            [sys.executable, "-m", "pytest", "--cov=app", "--cov-report=term-missing", "--cov-report=html", "-q"],
            capture_output=True,
            text=True
        )
        
        # Procurar porcentagem de cobertura
        for line in cov_result.stdout.split('\n'):
            if "TOTAL" in line and "%" in line:
                print(f"\n[COBERTURA] {line.strip()}")
                
                # Extrair porcentagem
                parts = line.split()
                for part in parts:
                    if part.endswith('%'):
                        try:
                            coverage = float(part.rstrip('%'))
                            if coverage >= 80:
                                print(f"\n[SUCESSO] META DE 80% ALCANÇADA! ({coverage}%)")
                            else:
                                print(f"\n[AVISO] Cobertura em {coverage}% (meta: 80%)")
                        except:
                            pass
        
        print("\n[INFO] Relatório HTML gerado em: htmlcov/index.html")
        
        return 0
    else:
        print("\n[AVISO] Ainda há testes falhando")
        print("\n[AÇÕES RECOMENDADAS]:")
        print("1. Verifique os erros específicos acima")
        print("2. Execute manualmente: python comprehensive-test-fix.py")
        print("3. Tente novamente: pytest tests/test_ecg_service_critical_coverage.py -vv")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
