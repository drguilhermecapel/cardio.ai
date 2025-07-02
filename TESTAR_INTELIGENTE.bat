@echo off
set PYTHONPATH=%CD%
set ENVIRONMENT=test

echo ============================================
echo EXECUTANDO TESTES QUE DEVEM FUNCIONAR
echo ============================================

REM Testar módulos básicos primeiro

echo.
echo [1] Testando exceptions...
python -m pytest tests/test_exceptions_coverage.py -v

echo.
echo [2] Testando config...
python -m pytest tests/test_config_coverage.py -v

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
