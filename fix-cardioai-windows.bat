@echo off
REM Script Batch para corrigir e executar testes do CardioAI Pro no Windows
REM Salve como: fix-cardioai.bat

echo ============================================================
echo CORRECAO AUTOMATICA CARDIOAI PRO - WINDOWS
echo ============================================================
echo.

REM Verificar se Python esta instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado! Instale Python 3.8 ou superior.
    pause
    exit /b 1
)

echo [OK] Python encontrado
echo.

REM Definir encoding UTF-8
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8

echo [INFO] Criando script de correcao temporario...

REM Criar script Python inline para corrigir encoding
echo import os > fix_temp.py
echo import sys >> fix_temp.py
echo from pathlib import Path >> fix_temp.py
echo. >> fix_temp.py
echo print("[INFO] Corrigindo arquivos...") >> fix_temp.py
echo. >> fix_temp.py
echo scripts = ['cardioai-fix-script.py', 'fix-validation-exception.py', >> fix_temp.py
echo            'fix-additional-issues.py', 'run-tests-cardioai.py', >> fix_temp.py
echo            'apply-all-fixes-python.py'] >> fix_temp.py
echo. >> fix_temp.py
echo for script in scripts: >> fix_temp.py
echo     if Path(script).exists(): >> fix_temp.py
echo         print(f"  Processando {script}...") >> fix_temp.py
echo         with open(script, 'r', encoding='utf-8', errors='ignore') as f: >> fix_temp.py
echo             content = f.read() >> fix_temp.py
echo         # Remover emojis >> fix_temp.py
echo         content = content.replace('✅', '[OK]') >> fix_temp.py
echo         content = content.replace('❌', '[ERRO]') >> fix_temp.py
echo         content = content.replace('⚠️', '[AVISO]') >> fix_temp.py
echo         content = content.replace('📌', '[INFO]') >> fix_temp.py
echo         content = content.replace('📊', '[STATS]') >> fix_temp.py
echo         content = content.replace('🎯', '[ALVO]') >> fix_temp.py
echo         content = content.replace('🚀', '[INICIO]') >> fix_temp.py
echo         content = content.replace('📦', '[PACOTE]') >> fix_temp.py
echo         content = content.replace('🔧', '[CONFIG]') >> fix_temp.py
echo         content = content.replace('📝', '[DOC]') >> fix_temp.py
echo         content = content.replace('🧹', '[LIMPAR]') >> fix_temp.py
echo         content = content.replace('🧪', '[TESTE]') >> fix_temp.py
echo         content = content.replace('📋', '[RELATORIO]') >> fix_temp.py
echo         content = content.replace('📁', '[DIR]') >> fix_temp.py
echo         content = content.replace('🎉', '[SUCESSO]') >> fix_temp.py
echo         content = content.replace('\u2705', '[OK]') >> fix_temp.py
echo         content = content.replace('\u274c', '[ERRO]') >> fix_temp.py
echo         content = content.replace('\U0001f527', '[CONFIG]') >> fix_temp.py
echo         with open(script, 'w', encoding='utf-8') as f: >> fix_temp.py
echo             f.write(content) >> fix_temp.py
echo print("[OK] Arquivos corrigidos!") >> fix_temp.py

REM Executar correção de encoding
python fix_temp.py
del fix_temp.py >nul 2>&1

echo.
echo [INFO] Instalando dependencias...
pip install pytest pytest-asyncio pytest-cov pytest-mock aiosqlite httpx numpy scipy

echo.
echo ============================================================
echo EXECUTANDO CORRECOES DO CARDIOAI
echo ============================================================
echo.

REM Executar scripts em ordem
echo [1/4] Executando correcoes principais...
python cardioai-fix-script.py
if errorlevel 1 echo [AVISO] Algumas correcoes falharam

echo.
echo [2/4] Corrigindo ValidationException...
python fix-validation-exception.py
if errorlevel 1 echo [AVISO] Correcao ValidationException falhou

echo.
echo [3/4] Executando correcoes adicionais...
python fix-additional-issues.py
if errorlevel 1 echo [AVISO] Correcoes adicionais falharam

echo.
echo [4/4] Executando testes...
python run-tests-cardioai.py

echo.
echo ============================================================
echo PROCESSO CONCLUIDO!
echo ============================================================
echo.
echo Para ver o relatorio de cobertura, abra:
echo   htmlcov\index.html
echo.
echo Para executar testes novamente:
echo   pytest --cov=app --cov-report=html
echo.

pause
