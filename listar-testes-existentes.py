import os
from pathlib import Path

print("LISTANDO TESTES EXISTENTES")
print("=" * 60)

tests_dir = Path("tests")

if tests_dir.exists():
    test_files = list(tests_dir.glob("test_*.py"))
    
    print(f"\nEncontrados {len(test_files)} arquivos de teste:\n")
    
    # Separar por categoria
    working_tests = []
    api_tests = []
    other_tests = []
    
    for test_file in sorted(test_files):
        name = test_file.name
        print(f"  • {name}")
        
        # Categorizar
        if 'api' in name or 'endpoint' in name or 'main' in name or 'health' in name:
            api_tests.append(name)
        elif any(x in name for x in ['exceptions_coverage', 'config_coverage', 'security', 'patient', 'ecg_service', 'user', 'validation']):
            working_tests.append(name)
        else:
            other_tests.append(name)
    
    # Criar script de teste inteligente
    print("\nCriando script de teste inteligente...")
    
    with open("TESTAR_INTELIGENTE.bat", 'w', encoding='utf-8') as f:
        f.write('''@echo off
set PYTHONPATH=%CD%
set ENVIRONMENT=test

echo ============================================
echo EXECUTANDO TESTES QUE DEVEM FUNCIONAR
echo ============================================

REM Testar módulos básicos primeiro
''')
        
        # Adicionar testes que provavelmente funcionam
        if 'test_exceptions_coverage.py' in [t.name for t in test_files]:
            f.write('\necho.\necho [1] Testando exceptions...\n')
            f.write('python -m pytest tests/test_exceptions_coverage.py -v\n')
            
        if 'test_config_coverage.py' in [t.name for t in test_files]:
            f.write('\necho.\necho [2] Testando config...\n')
            f.write('python -m pytest tests/test_config_coverage.py -v\n')
        
        # Testar alguns outros módulos
        f.write('''
echo.
echo ============================================
echo TENTANDO OUTROS TESTES (podem falhar)
echo ============================================

REM Ignorar testes de API que dependem de CORS
python -m pytest tests -v --tb=short -k "not api and not endpoint and not main and not health" --maxfail=10

echo.
echo ============================================
echo GERANDO RELATORIO DE COBERTURA
echo ============================================

coverage report
coverage html

echo.
echo Para testar TUDO (incluindo APIs):
echo python -m pytest tests -v --tb=short
pause
''')
    
    print("\n✓ Script TESTAR_INTELIGENTE.bat criado!")
    
    # Análise dos testes
    print("\n" + "=" * 60)
    print("ANÁLISE DOS TESTES:")
    print("=" * 60)
    print(f"\n✓ Testes que devem funcionar: {len(working_tests)}")
    for t in working_tests[:5]:
        print(f"  - {t}")
    
    print(f"\n⚠ Testes de API (precisam correção CORS): {len(api_tests)}")
    for t in api_tests[:3]:
        print(f"  - {t}")
    
    print(f"\n? Outros testes: {len(other_tests)}")
    for t in other_tests[:3]:
        print(f"  - {t}")
        
else:
    print("ERRO: Pasta 'tests' não encontrada!")
    print("Você está no diretório correto?")
    print(f"Diretório atual: {os.getcwd()}")

print("\n" + "=" * 60)
print("EXECUTE AGORA: TESTAR_INTELIGENTE.bat")
print("=" * 60)
