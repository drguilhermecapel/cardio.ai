@echo off
cd backend
set PYTHONPATH=%CD%
set ENVIRONMENT=test

echo.
echo ========================================
echo TESTANDO MODULOS FUNCIONAIS
echo ========================================

REM Testar apenas os módulos que devem funcionar
python -m pytest tests/test_auth_service.py tests/test_ecg_analysis_service.py tests/test_ecg_service.py tests/test_patient_service.py tests/test_security.py tests/test_user_service.py tests/test_validation_service.py -v --cov=app --cov-report=term-missing

echo.
echo ========================================
echo Se quiser testar TUDO (pode ter erros):
echo python -m pytest tests -v --tb=short
echo ========================================
pause
