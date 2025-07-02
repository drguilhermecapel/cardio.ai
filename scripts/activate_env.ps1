# Ativa ambiente virtual no Windows PowerShell
$venvPath = "C:\Users\User\OneDrive\Documentos\GitHub\cardio.ai.pro\venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    & $venvPath
    Write-Host "[OK] Ambiente virtual ativado!" -ForegroundColor Green
    Write-Host "[INFO] Diretorio do projeto: C:\Users\User\OneDrive\Documentos\GitHub\cardio.ai.pro" -ForegroundColor Cyan
} else {
    Write-Host "[ERRO] Ambiente virtual nao encontrado!" -ForegroundColor Red
    Write-Host "Execute primeiro: python scripts/setup_training_clean.py" -ForegroundColor Yellow
}
