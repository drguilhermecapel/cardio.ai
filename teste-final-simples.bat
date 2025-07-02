@echo off
set PYTHONPATH=%CD%
set ENVIRONMENT=test

cls
echo ============================================
echo CARDIOAI PRO - TESTE FINAL COM COBERTURA
echo ============================================
echo.
echo Executando 981 testes...
echo (Isso pode levar 2-3 minutos)
echo.

REM Executar sem parar em erros, mostrando apenas resumo
python -m pytest tests --cov=app --cov-report=term:skip-covered --cov-report=html -q --tb=short

echo.
echo ============================================
echo Para ver detalhes da cobertura:
echo Abra: htmlcov\index.html
echo ============================================
pause
