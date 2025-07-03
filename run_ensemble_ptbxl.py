@echo off
echo ===============================================
echo   Treinamento de Ensemble ECG com PTB-XL
echo ===============================================
echo.

REM Configurações - AJUSTE CONFORME NECESSÁRIO
set DATA_PATH=D:\ptb-xl\npy_lr
set MODELS=cnn resnet inception attention
set ENSEMBLE_METHOD=weighted_voting
set EPOCHS=100
set BATCH_SIZE=32
set OUTPUT_DIR=./ensemble_results

echo Configurações:
echo - Caminho dos dados: %DATA_PATH%
echo - Modelos: %MODELS%
echo - Método: %ENSEMBLE_METHOD%
echo - Épocas: %EPOCHS%
echo - Batch size: %BATCH_SIZE%
echo - Saída: %OUTPUT_DIR%
echo.

REM Verificar se Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ERRO: Python não encontrado!
    echo Por favor, instale Python 3.7 ou superior.
    pause
    exit /b 1
)

REM Criar diretório de saída se não existir
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

REM Executar o treinamento
echo Iniciando treinamento...
echo.

python ensemble_ecg_training.py ^
    --data-path %DATA_PATH% ^
    --models %MODELS% ^
    --ensemble-method %ENSEMBLE_METHOD% ^
    --epochs %EPOCHS% ^
    --batch-size %BATCH_SIZE% ^
    --output-dir %OUTPUT_DIR%

if errorlevel 1 (
    echo.
    echo ERRO: O treinamento falhou!
    echo Verifique os logs acima para mais detalhes.
) else (
    echo.
    echo Treinamento concluído com sucesso!
    echo Resultados salvos em: %OUTPUT_DIR%
)

echo.
pause
