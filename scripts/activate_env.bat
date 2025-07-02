@echo off
REM Ativa ambiente virtual no Windows CMD
set VENV_PATH=C:\Users\User\OneDrive\Documentos\GitHub\cardio.ai.pro\venv\Scripts\activate.bat
if exist "%VENV_PATH%" (
    call "%VENV_PATH%"
    echo [OK] Ambiente virtual ativado!
    echo [INFO] Diretorio do projeto: C:\Users\User\OneDrive\Documentos\GitHub\cardio.ai.pro
) else (
    echo [ERRO] Ambiente virtual nao encontrado!
    echo Execute primeiro: python scripts/setup_training_clean.py
)
