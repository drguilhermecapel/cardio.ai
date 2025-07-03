#!/usr/bin/env python3
"""Script seguro para executar testes do CardioAI Pro."""

import subprocess
import sys
import os

def run_tests():
    """Executa testes com configuração segura."""
    os.environ["ENVIRONMENT"] = "test"
    os.environ["PYTHONPATH"] = str(os.getcwd())
    
    print("🧪 Executando testes do CardioAI Pro...")
    print("=" * 60)
    
    # Comandos de teste em ordem de prioridade
    commands = [
        # Testes sem pyedflib primeiro
        ["pytest", "tests/test_exceptions_coverage.py", "-v"],
        ["pytest", "tests/test_config_coverage.py", "-v"],
        
        # Testes principais com coverage
        ["pytest", "--cov=app", "--cov-report=term-missing", "-v", "--tb=short"],
        
        # Relatório final
        ["coverage", "report"],
        ["coverage", "html"],
    ]
    
    for cmd in commands:
        print(f"\n📍 Executando: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=False, text=True)
            if result.returncode != 0:
                print(f"⚠️  Comando falhou com código {result.returncode}")
        except Exception as e:
            print(f"❌ Erro ao executar comando: {e}")
    
    print("\n✅ Testes concluídos!")
    print("📊 Relatório HTML disponível em: htmlcov/index.html")

if __name__ == "__main__":
    run_tests()
