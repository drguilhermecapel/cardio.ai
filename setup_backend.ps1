# backend/setup_backend.ps1
# Script de configuração do ambiente de desenvolvimento backend para Windows

Write-Host "🚀 CardioAI Pro - Configuração do Backend" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Verificar se está no diretório correto
if (-not (Test-Path "pyproject.toml")) {
    Write-Host "❌ Erro: Execute este script no diretório backend/" -ForegroundColor Red
    exit 1
}

# Função para executar comandos e verificar erros
function Execute-Command {
    param(
        [string]$Command,
        [string]$Description
    )
    
    Write-Host "📦 $Description..." -ForegroundColor Yellow
    try {
        Invoke-Expression $Command
        if ($LASTEXITCODE -ne 0) {
            Write-Host "⚠️ Aviso no comando: $Command" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "❌ Erro ao executar: $Command" -ForegroundColor Red
        Write-Host $_.Exception.Message -ForegroundColor Red
    }
}

# 1. Atualizar poetry.lock
Execute-Command "poetry lock --no-update" "Atualizando poetry.lock"

# 2. Instalar dependências principais
Execute-Command "poetry install" "Instalando dependências principais"

# 3. Adicionar ferramentas de desenvolvimento uma por uma
Write-Host "`n🛠️ Instalando ferramentas de desenvolvimento..." -ForegroundColor Green

$devTools = @(
    @{Name="pytest-cov"; Description="Cobertura de testes"},
    @{Name="pytest-asyncio"; Description="Testes assíncronos"},
    @{Name="pytest-mock"; Description="Mocking para testes"},
    @{Name="pytest-xdist"; Description="Execução paralela de testes"},
    @{Name="radon"; Description="Análise de complexidade"},
    @{Name="bandit[toml]"; Description="Verificação de segurança"},
    @{Name="pip-audit"; Description="Auditoria de dependências"},
    @{Name="black"; Description="Formatador de código"},
    @{Name="isort"; Description="Organizador de imports"},
    @{Name="flake8"; Description="Linter"},
    @{Name="mypy"; Description="Type checking"},
    @{Name="httpx"; Description="Cliente HTTP para testes"},
    @{Name="factory-boy"; Description="Factories para testes"},
    @{Name="faker"; Description="Gerador de dados fake"}
)

foreach ($tool in $devTools) {
    Execute-Command "poetry add --group dev $($tool.Name)" "Instalando $($tool.Description)"
}

# 4. Criar estrutura de diretórios
Write-Host "`n📁 Criando estrutura de diretórios..." -ForegroundColor Yellow

$directories = @(
    "tests/medical",
    "tests/unit",
    "tests/integration",
    "htmlcov",
    "scripts",
    ".github/workflows"
)

foreach ($dir in $directories) {
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        Write-Host "✅ Criado: $dir" -ForegroundColor Green
    }
}

# 5. Copiar arquivos de teste se existirem
Write-Host "`n📄 Configurando arquivos de teste..." -ForegroundColor Yellow

$testFiles = @{
    "test_ml_model_service_coverage.py" = "tests/"
    "test_medical_critical_components.py" = "tests/"
    "coverage_automation.py" = "scripts/"
}

foreach ($file in $testFiles.Keys) {
    if (Test-Path $file) {
        Copy-Item $file -Destination $testFiles[$file] -Force
        Write-Host "✅ Copiado: $file para $($testFiles[$file])" -ForegroundColor Green
    }
}

# 6. Verificar instalação
Write-Host "`n🔍 Verificando instalação..." -ForegroundColor Yellow

$commands = @(
    @{Cmd="poetry run pytest --version"; Name="pytest"},
    @{Cmd="poetry run coverage --version"; Name="coverage"},
    @{Cmd="poetry run radon --version"; Name="radon"},
    @{Cmd="poetry run bandit --version"; Name="bandit"},
    @{Cmd="poetry run pip-audit --version"; Name="pip-audit"}
)

$allOk = $true
foreach ($cmd in $commands) {
    try {
        $output = Invoke-Expression $cmd.Cmd 2>&1
        Write-Host "✅ $($cmd.Name): OK" -ForegroundColor Green
    } catch {
        Write-Host "❌ $($cmd.Name): FALHOU" -ForegroundColor Red
        $allOk = $false
    }
}

# 7. Executar teste inicial
if ($allOk) {
    Write-Host "`n🧪 Executando teste de cobertura inicial..." -ForegroundColor Yellow
    Execute-Command "poetry run pytest --cov=app --cov-report=term --cov-report=html -x" "Executando testes"
}

# 8. Resumo final
Write-Host "`n✨ Configuração concluída!" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Próximos passos:" -ForegroundColor Cyan
Write-Host "  1. Copie os arquivos de teste para tests/" -ForegroundColor White
Write-Host "  2. Execute: poetry run pytest --cov=app --cov-report=html" -ForegroundColor White
Write-Host "  3. Verifique cobertura em: htmlcov/index.html" -ForegroundColor White
Write-Host "  4. Execute testes críticos: poetry run pytest tests/test_medical_critical_components.py --cov-fail-under=100" -ForegroundColor White
Write-Host ""
Write-Host "🔧 Comandos úteis:" -ForegroundColor Cyan
Write-Host "  - Verificar segurança: poetry run pip-audit" -ForegroundColor White
Write-Host "  - Verificar complexidade: poetry run radon cc app -a" -ForegroundColor White
Write-Host "  - Formatar código: poetry run black app tests" -ForegroundColor White
Write-Host "  - Ordenar imports: poetry run isort app tests" -ForegroundColor White
Write-Host ""

# Perguntar se deseja abrir o relatório de cobertura
if (Test-Path "htmlcov/index.html") {
    $response = Read-Host "Deseja abrir o relatório de cobertura? (S/N)"
    if ($response -eq 'S' -or $response -eq 's') {
        Start-Process "htmlcov/index.html"
    }
}
