import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

import pytest
from unittest.mock import Mock, AsyncMock

def test_basic_imports():
    """Testa se todos os módulos podem ser importados."""
    try:
        from app.services.ecg_service import ECGAnalysisService
        from app.core.constants import FileType, ClinicalUrgency
        from app.core.exceptions import CardioAIException
        assert True
    except Exception as e:
        pytest.fail(f"Erro ao importar: {e}")

@pytest.mark.asyncio
async def test_ecg_service_methods():
    """Testa se ECGAnalysisService tem todos os métodos necessários."""
    from app.services.ecg_service import ECGAnalysisService
    
    mock_db = AsyncMock()
    service = ECGAnalysisService(mock_db)
    
    # Verificar métodos
    assert hasattr(service, 'get_analyses_by_patient')
    assert hasattr(service, 'delete_analysis')
    assert hasattr(service, 'search_analyses')
    assert hasattr(service, 'generate_report')
    assert hasattr(service, '_extract_measurements')
    assert hasattr(service, '_assess_clinical_urgency')
