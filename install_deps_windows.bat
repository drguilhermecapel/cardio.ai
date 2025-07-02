@echo off
REM Script para instalar dependências do Cardio.AI.Pro no Windows
echo ========================================
echo   Cardio.AI.Pro - Instalacao Manual
echo ========================================
echo.

REM Verifica se o ambiente virtual existe
if not exist "venv\Scripts\python.exe" (
    echo [ERRO] Ambiente virtual nao encontrado!
    echo Execute primeiro: python scripts\setup_training.py
    pause
    exit /b 1
)

REM Ativa o ambiente virtual
echo [1/4] Ativando ambiente virtual...
call venv\Scripts\activate.bat

REM Atualiza pip
echo [2/4] Atualizando pip...
python -m pip install --upgrade pip

REM Instala pacotes essenciais primeiro
echo [3/4] Instalando pacotes essenciais...
echo.

echo Instalando NumPy...
pip install numpy>=1.21.0

echo Instalando Pandas...
pip install pandas>=1.3.0

echo Instalando Scikit-learn...
pip install scikit-learn>=1.0.0

echo Instalando Matplotlib...
pip install matplotlib>=3.4.0

echo.
echo [4/4] Instalando todas as dependencias...
pip install -r requirements.txt

echo.
echo ========================================
echo   Instalacao Concluida!
echo ========================================
echo.
echo Proximos passos:
echo 1. Mantenha o ambiente virtual ativado
echo 2. Execute: python scripts\train_model.py
echo.
pause