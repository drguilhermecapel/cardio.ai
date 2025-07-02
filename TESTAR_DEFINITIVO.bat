@echo off
set PYTHONPATH=%CD%
set ENVIRONMENT=test

echo ============================================
echo EXECUTANDO TODOS OS TESTES COM COBERTURA
echo ============================================

REM Executar TODOS os testes, ignorando falhas
python -m pytest tests -v --tb=short --cov=app --cov-report=term-missing --cov-report=html --maxfail=50 -x

echo.
echo ============================================
echo RELATORIO DE COBERTURA
echo ============================================

coverage report

echo.
echo Relatorio HTML: htmlcov\index.html
echo.
echo Para executar testes especificos:
echo python -m pytest tests/test_ecg_service.py -v
pause
