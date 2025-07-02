import os
from pathlib import Path

print("CORRIGINDO ÚLTIMOS ERROS PARA EXECUTAR TODOS OS TESTES")
print("=" * 60)

# 1. Adicionar exceções faltantes
print("\n[1/3] Adicionando exceções faltantes...")
exceptions_file = Path("app/core/exceptions.py")

exceptions_to_add = '''

# Exceções Not Found específicas
class ECGNotFoundException(NotFoundException):
    """ECG not found exception."""
    def __init__(self, ecg_id: str) -> None:
        super().__init__(f"ECG {ecg_id} not found")

class PatientNotFoundException(NotFoundException):
    """Patient not found exception."""
    def __init__(self, patient_id: str) -> None:
        super().__init__(f"Patient {patient_id} not found")

class UserNotFoundException(NotFoundException):
    """User not found exception."""
    def __init__(self, user_id: str) -> None:
        super().__init__(f"User {user_id} not found")

class ConflictException(CardioAIException):
    """Exception for conflict errors."""
    def __init__(self, message: str) -> None:
        super().__init__(message, "CONFLICT_ERROR", 409)

class PermissionDeniedException(CardioAIException):
    """Exception for permission denied errors."""
    def __init__(self, message: str) -> None:
        super().__init__(message, "PERMISSION_DENIED", 403)

class MLModelException(CardioAIException):
    """ML Model exception."""
    def __init__(self, message: str, model_name: str = None) -> None:
        details = {"model_name": model_name} if model_name else {}
        super().__init__(message, "ML_MODEL_ERROR", 500, details)

class ValidationNotFoundException(NotFoundException):
    """Validation not found."""
    def __init__(self, validation_id: str) -> None:
        super().__init__(f"Validation {validation_id} not found")

class AnalysisNotFoundException(NotFoundException):
    """Analysis not found."""
    def __init__(self, analysis_id: str) -> None:
        super().__init__(f"Analysis {analysis_id} not found")

class ValidationAlreadyExistsException(ConflictException):
    """Validation already exists."""
    def __init__(self, analysis_id: str) -> None:
        super().__init__(f"Validation for analysis {analysis_id} already exists")

class InsufficientPermissionsException(PermissionDeniedException):
    """Insufficient permissions."""
    def __init__(self, required_permission: str) -> None:
        super().__init__(f"Insufficient permissions. Required: {required_permission}")

class RateLimitExceededException(CardioAIException):
    """Rate limit exceeded."""
    def __init__(self, message: str = "Rate limit exceeded") -> None:
        super().__init__(message, "RATE_LIMIT_EXCEEDED", 429)

class FileProcessingException(CardioAIException):
    """File processing error."""
    def __init__(self, message: str, filename: str = None) -> None:
        details = {"filename": filename} if filename else {}
        super().__init__(message, "FILE_PROCESSING_ERROR", 422, details)

class DatabaseException(CardioAIException):
    """Database error."""
    def __init__(self, message: str) -> None:
        super().__init__(message, "DATABASE_ERROR", 500)

class ExternalServiceException(CardioAIException):
    """External service error."""
    def __init__(self, message: str, service_name: str = None) -> None:
        details = {"service_name": service_name} if service_name else {}
        super().__init__(message, "EXTERNAL_SERVICE_ERROR", 502, details)

class NonECGImageException(CardioAIException):
    """Non-ECG image detected."""
    def __init__(self) -> None:
        super().__init__("Uploaded image is not an ECG", "NON_ECG_IMAGE", 422)
'''

if exceptions_file.exists():
    with open(exceptions_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Adicionar apenas se não existir
    if "ECGNotFoundException" not in content:
        with open(exceptions_file, 'a', encoding='utf-8') as f:
            f.write(exceptions_to_add)
        print("  ✓ Todas as exceções adicionadas!")
    else:
        print("  ✓ Exceções já existem!")

# 2. Corrigir config.py - adicionar atributos faltantes
print("\n[2/3] Corrigindo configurações...")
config_file = Path("app/core/config.py")

if config_file.exists():
    with open(config_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Adicionar ML_MODEL_PATH e outros
    additions_needed = []
    if "ML_MODEL_PATH" not in content:
        additions_needed.append('    ML_MODEL_PATH: str = "models"')
    if "ML_BATCH_SIZE" not in content:
        additions_needed.append('    ML_BATCH_SIZE: int = 32')
    if "ML_MAX_QUEUE_SIZE" not in content:
        additions_needed.append('    ML_MAX_QUEUE_SIZE: int = 1000')
    if "UPLOAD_PATH" not in content:
        additions_needed.append('    UPLOAD_PATH: str = "uploads"')
    if "MAX_UPLOAD_SIZE" not in content:
        additions_needed.append('    MAX_UPLOAD_SIZE: int = 104857600  # 100MB')
    
    if additions_needed:
        # Encontrar onde adicionar (dentro da classe Settings)
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'class Settings' in line:
                # Encontrar o final da classe ou um bom lugar para inserir
                j = i + 1
                while j < len(lines) and (lines[j].startswith('    ') or not lines[j].strip()):
                    j += 1
                
                # Inserir antes do fim da classe
                for k in range(j-1, i, -1):
                    if lines[k].strip() and lines[k].startswith('    '):
                        for addition in reversed(additions_needed):
                            lines.insert(k+1, addition)
                        break
                break
        
        content = '\n'.join(lines)
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("  ✓ Configurações adicionadas!")

# 3. Corrigir constants.py - adicionar PATIENT em UserRoles
print("\n[3/3] Corrigindo constantes...")
constants_file = Path("app/core/constants.py")

if constants_file.exists():
    with open(constants_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Adicionar PATIENT ao UserRoles se não existir
    if "PATIENT" not in content:
        # Encontrar UserRoles enum
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'class UserRoles' in line:
                # Adicionar PATIENT após outros roles
                j = i + 1
                while j < len(lines) and (lines[j].startswith('    ') or not lines[j].strip()):
                    if 'TECHNICIAN' in lines[j] or 'PHYSICIAN' in lines[j]:
                        # Adicionar após este
                        lines.insert(j+1, '    PATIENT = "patient"')
                        break
                    j += 1
                break
        
        content = '\n'.join(lines)
        with open(constants_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("  ✓ PATIENT role adicionado!")

print("\n" + "=" * 60)
print("CORREÇÕES FINAIS APLICADAS!")
print("=" * 60)

# Criar script de teste definitivo
print("\nCriando script de teste definitivo...")
with open("TESTAR_DEFINITIVO.bat", 'w', encoding='utf-8') as f:
    f.write('''@echo off
set PYTHONPATH=%CD%
set ENVIRONMENT=test

echo ============================================
echo EXECUTANDO TODOS OS TESTES COM COBERTURA
echo ============================================

REM Executar TODOS os testes, ignorando falhas
python -m pytest tests -v --tb=short --cov=app --cov-report=term-missing --cov-report=html --maxfail=50 -x

echo.
echo ============================================
echo RELATORIO DE COBERTURA
echo ============================================

coverage report

echo.
echo Relatorio HTML: htmlcov\index.html
echo.
echo Para executar testes especificos:
echo python -m pytest tests/test_ecg_service.py -v
pause
''')

print("\nPRONTO! Execute:")
print("  TESTAR_DEFINITIVO.bat")
print("\nOu para ver progresso gradual:")
print("  python -m pytest tests/test_ecg_service.py -v")
print("  python -m pytest tests/test_patient_service.py -v")
print("  python -m pytest tests/test_validation_service.py -v")
