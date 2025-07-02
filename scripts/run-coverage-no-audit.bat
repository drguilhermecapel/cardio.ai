@echo off
REM Script para executar cobertura ignorando avisos de audit

echo =====================================
echo   CardioAI Pro - Cobertura de Testes
echo =====================================
echo.

REM Verificar instalação
if not exist "node_modules" (
    echo [!] Instalando dependencias...
    call npm install --legacy-peer-deps
)

echo [1/2] Executando teste simples de verificacao...
call npx vitest run src/simple.test.ts --reporter=verbose

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERRO] Teste basico falhou!
    echo.
    echo Tente executar:
    echo   npm install --force
    echo   npm run test:coverage
    pause
    exit /b 1
)

echo.
echo [OK] Teste basico passou!
echo.

echo [2/2] Gerando relatorio de cobertura completo...
call npm run test:coverage

if %ERRORLEVEL% EQU 0 (
    echo.
    echo =====================================
    echo   SUCESSO! Cobertura gerada!
    echo =====================================
    echo.
    
    if exist "coverage\lcov-report\index.html" (
        echo Abrindo relatorio no navegador...
        start coverage\lcov-report\index.html
    )
    
    echo Relatorios disponiveis em:
    echo - HTML: coverage\lcov-report\index.html
    echo - JSON: coverage\coverage-final.json
    echo - LCOV: coverage\lcov.info
) else (
    echo.
    echo [AVISO] Cobertura pode ter sido gerada mesmo com avisos
    echo Verifique a pasta coverage\
)

echo.
pause
