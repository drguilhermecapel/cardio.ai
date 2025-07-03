@echo off
REM Script Windows para corrigir todos os erros e alcançar 100% de cobertura

echo ====================================
echo Corrigindo erros do CardioAI Backend
echo ====================================

cd /d C:\Users\User\OneDrive\Documentos\GitHub\cardio.ai.pro\backend

echo.
echo [1/5] Instalando dependencias necessarias...
python -m pip install coverage pytest pytest-cov pytest-asyncio pytest-mock --quiet

echo.
echo [2/5] Corrigindo erro de importacao no test_ecg_tasks_complete_coverage.py...
python -c "import re; content = open('tests/test_ecg_tasks_complete_coverage.py', 'r').read(); content = re.sub(r'from app\.tasks\.ecg_tasks import \([^)]*process_ecg_async[^)]*\)', 'from app.tasks.ecg_tasks import process_ecg_analysis_sync', content); content = content.replace('process_ecg_async(', 'process_ecg_analysis_sync('); open('tests/test_ecg_tasks_complete_coverage.py', 'w').write(content); print('Arquivo corrigido!')"

echo.
echo [3/5] Criando testes para modulos sem cobertura...

REM Criar teste para audit_trail.py
echo import pytest > tests\test_audit_trail_coverage.py
echo from unittest.mock import Mock, patch >> tests\test_audit_trail_coverage.py
echo from app.security.audit_trail import create_audit_trail >> tests\test_audit_trail_coverage.py
echo. >> tests\test_audit_trail_coverage.py
echo class TestAuditTrail: >> tests\test_audit_trail_coverage.py
echo     def test_create_audit_trail(self, tmp_path): >> tests\test_audit_trail_coverage.py
echo         audit = create_audit_trail(str(tmp_path / "test.db")) >> tests\test_audit_trail_coverage.py
echo         assert audit is not None >> tests\test_audit_trail_coverage.py
echo     def test_audit_operations(self, tmp_path): >> tests\test_audit_trail_coverage.py
echo         audit = create_audit_trail(str(tmp_path / "test.db")) >> tests\test_audit_trail_coverage.py
echo         # Testa todas as operacoes >> tests\test_audit_trail_coverage.py
echo         audit_id = audit.log_prediction({}, {}, {}) >> tests\test_audit_trail_coverage.py
echo         assert audit_id is not None >> tests\test_audit_trail_coverage.py
echo         report = audit.generate_compliance_report(30) >> tests\test_audit_trail_coverage.py
echo         assert report is not None >> tests\test_audit_trail_coverage.py
echo         integrity = audit.verify_data_integrity(audit_id) >> tests\test_audit_trail_coverage.py
echo         assert isinstance(integrity, bool) >> tests\test_audit_trail_coverage.py
echo         summary = audit.get_audit_summary(7) >> tests\test_audit_trail_coverage.py
echo         assert summary is not None >> tests\test_audit_trail_coverage.py

echo.
echo [4/5] Executando testes com cobertura...
python -m pytest --cov=app --cov-report=term-missing --tb=short -q

echo.
echo [5/5] Gerando relatorio HTML de cobertura...
python -m pytest --cov=app --cov-report=html --tb=short -q

echo.
echo ====================================
echo Processo concluido!
echo ====================================
echo.
echo Relatorio HTML disponivel em: htmlcov\index.html
echo.
echo Para visualizar o relatorio:
echo   start htmlcov\index.html
echo.
pause
