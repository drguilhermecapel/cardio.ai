#!/usr/bin/env python3
"""
Script para executar os testes do CardioAI Pro após as correções
"""

import os
import sys
import subprocess
from pathlib import Path
import json

BACKEND_DIR = Path.cwd() / "backend" if (Path.cwd() / "backend").exists() else Path.cwd()

def run_command(cmd, cwd=None):
    """Executa um comando e retorna o resultado."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd or BACKEND_DIR
        )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def install_test_dependencies():
    """Instala dependências necessárias para os testes."""
    print("[PACOTE] Instalando dependências de teste...")
    
    dependencies = [
        "pytest>=7.0.0",
        "pytest-asyncio>=0.21.0", 
        "pytest-cov>=4.0.0",
        "pytest-mock>=3.10.0",
        "aiosqlite>=0.19.0",
        "httpx>=0.24.0"
    ]
    
    for dep in dependencies:
        success, _, _ = run_command(f"pip install {dep}")
        if success:
            print(f"  [OK] {dep}")
        else:
            print(f"  [ERRO] {dep}")


def run_critical_tests():
    """Executa apenas os testes críticos."""
    print("\n[ALVO] Executando testes CRÍTICOS...")
    
    cmd = "pytest tests/test_ecg_service_critical_coverage.py -v --tb=short"
    success, stdout, stderr = run_command(cmd)
    
    if success:
        print("[OK] Testes críticos passaram!")
    else:
        print("[ERRO] Testes críticos falharam:")
        print(stdout)
        print(stderr)
    
    return success


def run_all_tests_with_coverage():
    """Executa todos os testes com relatório de cobertura."""
    print("\n[STATS] Executando TODOS os testes com cobertura...")
    
    cmd = "pytest --cov=app --cov-report=term-missing --cov-report=html --cov-report=json -v"
    success, stdout, stderr = run_command(cmd)
    
    # Extrair informações de cobertura
    coverage_line = None
    for line in stdout.split('\n'):
        if 'TOTAL' in line and '%' in line:
            coverage_line = line
            break
    
    if coverage_line:
        try:
            # Extrair porcentagem de cobertura
            parts = coverage_line.split()
            for part in parts:
                if part.endswith('%'):
                    coverage = float(part.rstrip('%'))
                    print(f"\n[STATS] Cobertura Total: {coverage}%")
                    
                    if coverage >= 80:
                        print("[OK] Meta de 80% alcançada!")
                    else:
                        print(f"[ERRO] Cobertura abaixo de 80% (atual: {coverage}%)")
                    break
        except:
            pass
    
    return success


def run_specific_test_files():
    """Executa arquivos de teste específicos que costumam falhar."""
    print("\n🔍 Executando testes específicos problemáticos...")
    
    test_files = [
        "tests/test_ecg_service.py",
        "tests/test_ecg_preprocessing_comprehensive.py",
        "tests/test_api_endpoints_full.py",
        "tests/test_core_services_comprehensive.py",
        "tests/test_missing_coverage_areas.py"
    ]
    
    results = {}
    
    for test_file in test_files:
        if (BACKEND_DIR / test_file).exists():
            cmd = f"pytest {test_file} -v --tb=short"
            success, stdout, stderr = run_command(cmd)
            
            # Contar testes passados/falhados
            passed = stdout.count(" PASSED")
            failed = stdout.count(" FAILED")
            errors = stdout.count(" ERROR")
            
            results[test_file] = {
                "success": success,
                "passed": passed,
                "failed": failed,
                "errors": errors
            }
            
            status = "[OK]" if success else "[ERRO]"
            print(f"{status} {test_file}: {passed} passed, {failed} failed, {errors} errors")
    
    return results


def generate_test_report():
    """Gera relatório final dos testes."""
    print("\n[DOC] RELATÓRIO FINAL DOS TESTES")
    print("=" * 60)
    
    # Verificar se existe relatório de cobertura em JSON
    coverage_json = BACKEND_DIR / "coverage.json"
    if coverage_json.exists():
        with open(coverage_json, 'r') as f:
            coverage_data = json.load(f)
            total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
            print(f"Cobertura Total: {total_coverage:.2f}%")
    
    # Sugestões de próximos passos
    print("\n[RELATORIO] Próximos Passos:")
    print("1. Se a cobertura está abaixo de 80%:")
    print("   - Verifique o relatório HTML: htmlcov/index.html")
    print("   - Identifique arquivos com baixa cobertura")
    print("   - Adicione testes para as linhas não cobertas")
    print("\n2. Se os testes críticos falham:")
    print("   - Verifique os logs de erro específicos")
    print("   - Execute: pytest tests/test_ecg_service_critical_coverage.py -vv")
    print("   - Ajuste as correções conforme necessário")
    print("\n3. Para debug detalhado:")
    print("   - pytest -vv --tb=long <arquivo_de_teste>")
    print("   - pytest --pdb <arquivo_de_teste>  # Para debug interativo")


def main():
    """Função principal."""
    os.chdir(BACKEND_DIR)
    
    print("[INICIO] EXECUTANDO TESTES DO CARDIOAI PRO")
    print(f"[DIR] Diretório: {BACKEND_DIR}")
    print("=" * 60)
    
    # 1. Instalar dependências
    install_test_dependencies()
    
    # 2. Executar testes críticos
    critical_success = run_critical_tests()
    
    # 3. Executar todos os testes com cobertura
    coverage_success = run_all_tests_with_coverage()
    
    # 4. Executar testes específicos
    specific_results = run_specific_test_files()
    
    # 5. Gerar relatório
    generate_test_report()
    
    # Retornar sucesso geral
    return critical_success and coverage_success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
