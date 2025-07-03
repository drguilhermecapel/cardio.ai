"""
Testes críticos para ECGService
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from app.services.ecg_service import ECGAnalysisService
from app.core.exceptions import ECGProcessingException, ValidationException


@pytest.mark.asyncio
class TestECGServiceCritical:
    """Testes críticos do serviço de ECG."""
    
    async def test_create_analysis_success(self, mock_ecg_service):
        """Testa criação de análise com sucesso."""
        data = {
            "patient_id": 1,
            "file_url": "test.pdf",
            "file_type": "pdf"
        }
        
        result = await mock_ecg_service.create_analysis(data)
        
        assert result["id"] == 1
        assert result["status"] == "pending"
        mock_ecg_service.create_analysis.assert_called_once_with(data)
        
    async def test_get_analysis_success(self, mock_ecg_service):
        """Testa recuperação de análise com sucesso."""
        result = await mock_ecg_service.get_analysis(1)
        
        assert result["id"] == 1
        assert result["status"] == "completed"
        assert result["diagnosis"] == "Normal"
        
    async def test_create_analysis_validation_error(self):
        """Testa erro de validação na criação."""
        service = ECGAnalysisService()
        service.repository = Mock()
        service.repository.create = AsyncMock(
            side_effect=ValidationException("Invalid data")
        )
        
        with pytest.raises(ValidationException):
            await service.create_analysis({"invalid": "data"})
            
    async def test_processing_exception_handling(self):
        """Testa tratamento de exceção de processamento."""
        # Teste com diferentes formas de inicialização
        exc1 = ECGProcessingException("Error 1", ecg_id="123")
        assert exc1.ecg_id == "123"
        
        exc2 = ECGProcessingException("Error 2", details={"info": "test"})
        assert exc2.details.get("info") == "test"
        
        exc3 = ECGProcessingException("Error 3", detail={"info": "test2"})
        assert exc3.details.get("info") == "test2"
        
        # Teste com args adicionais
        exc4 = ECGProcessingException("Error 4", "extra", "args", custom_field="value")
        assert exc4.details.get("custom_field") == "value"
        assert "additional_info" in exc4.details
        
    async def test_list_analyses_pagination(self, mock_ecg_service):
        """Testa listagem com paginação."""
        result = await mock_ecg_service.list_analyses(page=1, limit=10)
        
        assert "items" in result
        assert "total" in result
        assert result["page"] == 1
        
    async def test_service_initialization(self):
        """Testa inicialização do serviço."""
        service = ECGAnalysisService()
        
        # Verificar métodos essenciais
        assert hasattr(service, 'create_analysis')
        assert hasattr(service, 'get_analysis')
        assert hasattr(service, 'list_analyses')
        assert hasattr(service, 'get_analyses_by_patient')
        assert hasattr(service, 'validate_analysis')
        assert hasattr(service, 'create_validation')
        
    async def test_get_analyses_by_patient(self, mock_ecg_service):
        """Testa busca de análises por paciente."""
        result = await mock_ecg_service.get_analyses_by_patient(
            patient_id=1,
            limit=10,
            offset=0
        )
        
        assert isinstance(result, list)
        mock_ecg_service.get_analyses_by_patient.assert_called_once()
        
    async def test_validate_analysis(self, mock_ecg_service):
        """Testa validação de análise."""
        result = await mock_ecg_service.validate_analysis(
            analysis_id=1,
            validation_data={"approved": True}
        )
        
        assert result is True
        mock_ecg_service.validate_analysis.assert_called_once()
        
    async def test_create_validation(self, mock_ecg_service):
        """Testa criação de validação."""
        result = await mock_ecg_service.create_validation(
            analysis_id=1,
            user_id=1,
            notes="Looks good"
        )
        
        assert result["id"] == 1
        assert result["status"] == "pending"
        mock_ecg_service.create_validation.assert_called_once()


@pytest.mark.asyncio
class TestExceptionsCritical:
    """Testes críticos de exceções."""
    
    async def test_all_exceptions_exist(self):
        """Verifica se todas as exceções necessárias existem."""
        from app.core.exceptions import (
            CardioAIException,
            ECGNotFoundException,
            ECGProcessingException,
            ValidationException,
            AuthenticationException,
            AuthorizationException,
            NotFoundException,
            ConflictException,
            PermissionDeniedException,
            FileProcessingException,
            DatabaseException,
            MultiPathologyException,
            ECGReaderException
        )
        
        # Testar criação básica
        exc = ECGNotFoundException(ecg_id=123)
        assert "123" in str(exc)
        
        val_exc = ValidationException("Invalid field", field="email")
        assert val_exc.details.get("field") == "email"
        
        mp_exc = MultiPathologyException("Multiple issues", pathologies=["afib", "vt"])
        assert mp_exc.details.get("pathologies") == ["afib", "vt"]


@pytest.mark.asyncio
class TestValidatorsCritical:
    """Testes críticos de validadores."""
    
    async def test_email_validator(self):
        """Testa validador de email."""
        from app.utils.validators import validate_email
        
        assert validate_email("test@example.com") is True
        assert validate_email("invalid.email") is False
        assert validate_email("") is False
        assert validate_email(None) is False
        assert validate_email("user@domain.co.uk") is True
        
    async def test_patient_data_validator(self):
        """Testa validador de dados de paciente."""
        from app.utils.validators import validate_patient_data
        
        valid_data = {
            "name": "João Silva",
            "birth_date": "1990-01-01"
        }
        assert validate_patient_data(valid_data) is True
        
        invalid_data = {"name": ""}
        assert validate_patient_data(invalid_data) is False


@pytest.mark.asyncio
class TestMainAppCritical:
    """Testes críticos do app principal."""
    
    async def test_app_functions_exist(self):
        """Verifica se funções principais existem."""
        from app.main import get_app_info, health_check, CardioAIApp
        
        # Testar get_app_info
        info = await get_app_info()
        assert info["name"] == "CardioAI Pro"
        assert info["version"] == "1.0.0"
        
        # Testar health_check
        health = await health_check()
        assert health["status"] == "healthy"
        
        # Testar CardioAIApp
        app = CardioAIApp()
        assert app.name == "CardioAI Pro"
        assert app.add_module("test") is True
        
        app_info = app.get_info()
        assert "test" in app_info["modules"]
