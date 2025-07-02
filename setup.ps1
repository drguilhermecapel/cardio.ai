# setup.ps1 - Script de setup automático
Write-Host "🚀 CardioAI Pro - Setup Automático do Frontend" -ForegroundColor Cyan
Write-Host ""

# Verificar Node.js
$nodeVersion = node --version
Write-Host "✓ Node.js: $nodeVersion" -ForegroundColor Green

# Instalar dependências
Write-Host ""
Write-Host "📦 Instalando dependências..." -ForegroundColor Yellow
npm install

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Instalação concluída!" -ForegroundColor Green
    
    # Verificar Vitest
    Write-Host ""
    Write-Host "🧪 Verificando Vitest..." -ForegroundColor Yellow
    npx vitest --version
    
    # Executar teste simples
    Write-Host ""
    Write-Host "🔍 Executando teste de verificação..." -ForegroundColor Yellow
    npm test -- --run --reporter=verbose
    
    Write-Host ""
    Write-Host "🎉 Setup completo! Você pode usar:" -ForegroundColor Green
    Write-Host "  npm run dev          - Iniciar desenvolvimento" -ForegroundColor Cyan
    Write-Host "  npm test            - Executar testes" -ForegroundColor Cyan
    Write-Host "  npm run test:coverage - Executar com cobertura" -ForegroundColor Cyan
} else {
    Write-Host ""
    Write-Host "❌ Erro na instalação!" -ForegroundColor Red
    Write-Host "Tente executar:" -ForegroundColor Yellow
    Write-Host "  npm cache clean --force" -ForegroundColor Cyan
    Write-Host "  npm install" -ForegroundColor Cyan
}
