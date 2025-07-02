@echo off
REM Script para executar cobertura de testes

echo =====================================
echo   CardioAI Pro - Teste de Cobertura
echo =====================================
echo.

REM Verificar se node_modules existe
if not exist "node_modules" (
    echo [ERRO] node_modules nao encontrado!
    echo Execute: npm install
    pause
    exit /b 1
)

REM Executar teste simples primeiro
echo [1/3] Executando teste simples...
npx vitest run src/simple.test.ts --reporter=verbose

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERRO] Teste simples falhou!
    echo Verifique a configuracao do Vitest.
    pause
    exit /b 1
)

echo.
echo [OK] Teste simples passou!
echo.

REM Executar todos os testes
echo [2/3] Executando todos os testes...
npm run test -- --run

echo.
echo [3/3] Gerando relatorio de cobertura...
npm run test:coverage

if %ERRORLEVEL% EQU 0 (
    echo.
    echo =====================================
    echo   SUCESSO! Cobertura gerada!
    echo =====================================
    echo.
    
    REM Tentar abrir o relatório
    if exist "coverage\lcov-report\index.html" (
        echo Abrindo relatorio...
        start coverage\lcov-report\index.html
    ) else (
        echo Relatorio salvo em: coverage\
    )
) else (
    echo.
    echo [ERRO] Falha ao gerar cobertura!
)

echo.
pause
