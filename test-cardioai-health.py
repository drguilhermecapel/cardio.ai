#!/usr/bin/env python3
"""
CardioAI Pro - Verificador de Saúde do Sistema
Testa se as correções funcionaram
"""

import sys
import os
from pathlib import Path

# Adicionar diretório ao path
sys.path.insert(0, str(Path(__file__).parent))

def test_system_health():
    """Verifica a saúde do sistema CardioAI"""
    print("=" * 60)
    print("🏥 CARDIOAI PRO - VERIFICAÇÃO DE SAÚDE")
    print("=" * 60)
    
    tests_passed = 0
    tests_failed = 0
    
    # Teste 1: Estrutura de diretórios
    print("\n1️⃣ Verificando estrutura de diretórios...")
    required_dirs = [
        "app",
        "app/core", 
        "app/services",
        "app/db",
        "tests"
    ]
    
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"   ✅ {dir_path}")
            tests_passed += 1
        else:
            print(f"   ❌ {dir_path} - NÃO ENCONTRADO")
            tests_failed += 1
            
    # Teste 2: Arquivos críticos
    print("\n2️⃣ Verificando arquivos críticos...")
    critical_files = [
        ("app/core/config.py", "Configuração"),
        ("app/services/ecg_service.py", "Serviço ECG"),
        ("app/db/base.py", "Base de Dados"),
        (".env", "Variáveis de Ambiente")
    ]
    
    for file_path, name in critical_files:
        if Path(file_path).exists():
            print(f"   ✅ {name}: {file_path}")
            tests_passed += 1
        else:
            print(f"   ❌ {name}: {file_path} - NÃO ENCONTRADO")
            tests_failed += 1
            
    # Teste 3: Importações
    print("\n3️⃣ Testando importações críticas...")
    
    # Config
    try:
        from app.core.config import settings
        print("   ✅ Config: Importado com sucesso")
        print(f"      - PROJECT_NAME: {settings.PROJECT_NAME}")
        print(f"      - ENVIRONMENT: {settings.ENVIRONMENT}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Config: ERRO - {str(e)[:100]}...")
        tests_failed += 1
        
    # ECG Service
    try:
        from app.services.ecg_service import ECGAnalysisService
        service = ECGAnalysisService()
        print("   ✅ ECGAnalysisService: Importado e instanciado")
        tests_passed += 1
    except SyntaxError as e:
        print(f"   ❌ ECGAnalysisService: ERRO DE SINTAXE - {e}")
        print(f"      Arquivo: {e.filename}")
        print(f"      Linha: {e.lineno}")
        tests_failed += 1
    except Exception as e:
        print(f"   ❌ ECGAnalysisService: ERRO - {str(e)[:100]}...")
        tests_failed += 1
        
    # Database
    try:
        from app.db.base import Base
        print("   ✅ Database Base: Importado com sucesso")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Database Base: ERRO - {str(e)[:100]}...")
        tests_failed += 1
        
    # Teste 4: Verificar sintaxe do ecg_service.py
    print("\n4️⃣ Verificando sintaxe do ecg_service.py...")
    ecg_file = Path("app/services/ecg_service.py")
    
    if ecg_file.exists():
        try:
            with open(ecg_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Procurar por problemas conhecidos
            if 'pending"}' in content and 'pending"}"' not in content:
                print("   ⚠️ AVISO: String 'pending\"}' encontrada - pode causar erro de sintaxe")
                
                # Mostrar contexto
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'pending"}' in line:
                        print(f"      Linha {i+1}: {line.strip()}")
                        
                tests_failed += 1
            else:
                print("   ✅ Nenhum problema de sintaxe conhecido detectado")
                tests_passed += 1
                
        except Exception as e:
            print(f"   ❌ Erro ao ler arquivo: {e}")
            tests_failed += 1
    else:
        print("   ❌ Arquivo não encontrado")
        tests_failed += 1
        
    # Teste 5: Pytest básico
    print("\n5️⃣ Executando teste pytest básico...")
    import subprocess
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--version"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"   ✅ Pytest instalado: {result.stdout.strip()}")
        tests_passed += 1
    else:
        print("   ❌ Pytest não instalado ou não funcional")
        tests_failed += 1
        
    # Relatório Final
    print("\n" + "=" * 60)
    print("📊 RELATÓRIO DE SAÚDE")
    print("=" * 60)
    
    total_tests = tests_passed + tests_failed
    health_percentage = (tests_passed / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\nTestes aprovados: {tests_passed}/{total_tests}")
    print(f"Saúde do sistema: {health_percentage:.1f}%")
    
    if health_percentage >= 80:
        print("\n✅ Sistema SAUDÁVEL! Pronto para executar testes.")
        print("\nPróximo comando:")
        print("   python -m pytest --cov=app --cov-report=html")
    elif health_percentage >= 50:
        print("\n⚠️ Sistema PARCIALMENTE FUNCIONAL.")
        print("\nRecomendação:")
        print("   Execute: python cardioai_defibrillator.py")
    else:
        print("\n❌ Sistema em ESTADO CRÍTICO!")
        print("\nAção urgente:")
        print("   Execute: fix_cardioai_now.bat")
        
    return health_percentage >= 80


if __name__ == "__main__":
    test_system_health()
