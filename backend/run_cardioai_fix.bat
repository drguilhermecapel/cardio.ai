@echo off
REM ============================================================
REM CardioAI Pro - Script de Correção Automática
REM Objetivo: Atingir 80% de cobertura de código
REM ============================================================

echo.
echo ============================================================
echo    CardioAI Pro - Correcao Automatica e Cobertura 80%%
echo ============================================================
echo.

REM Verificar se Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado! Instale Python 3.8 ou superior.
    echo        Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM Verificar se estamos no diretório correto
if not exist "app\" (
    echo [ERRO] Execute este script no diretorio backend!
    echo        Diretorio atual: %CD%
    pause
    exit /b 1
)

echo [INFO] Iniciando processo de correcao...
echo.

REM Passo 1: Verificar status inicial
echo ========================================
echo PASSO 0: Verificar Status Inicial
echo ========================================
python verify_system_status.py
if errorlevel 1 (
    echo [AVISO] Problemas detectados no sistema
)
echo.
pause

REM Passo 2: Executar correção completa
echo ========================================
echo PASSO PRINCIPAL: Executar Correcao Completa
echo ========================================
python execute_complete_fix.py
if errorlevel 1 (
    echo [ERRO] Falha na execucao da correcao completa
    pause
    exit /b 1
)

echo.
echo ========================================
echo PROCESSO CONCLUIDO!
echo ========================================
echo.

REM Verificar status final
python verify_system_status.py

echo.
echo [INFO] Para ver o relatorio de cobertura:
echo        start htmlcov\index.html
echo.
echo [INFO] Para executar o servidor:
echo        uvicorn app.main:app --reload
echo.

pause