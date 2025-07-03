"""
Configuração global para testes
"""
import os
import sys
import asyncio
from typing import AsyncGenerator, Generator
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from unittest.mock import Mock, AsyncMock

# Adicionar diretório raiz ao path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configurar ambiente de teste
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["JWT_SECRET_KEY"] = "test-secret-key"
os.environ["JWT_ALGORITHM"] = "HS256"
os.environ["BACKEND_CORS_ORIGINS"] = '["http://localhost:3000"]'

# Importar após configurar ambiente
from app.core.database import Base
from app.core.config import settings


@pytest.fixture(scope="session")
def event_loop() -> Generator:
    """Cria event loop para toda a sessão de testes."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def db_engine():
    """Cria engine de banco de dados para testes."""
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True
    )
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """Cria sessão de banco de dados para testes."""
    async_session_maker = async_sessionmaker(
        db_engine, 
        class_=AsyncSession, 
        expire_on_commit=False
    )
    
    async with async_session_maker() as session:
        yield session
        await session.rollback()


@pytest.fixture
def mock_ecg_service():
    """Mock do serviço de ECG."""
    service = AsyncMock()
    service.create_analysis = AsyncMock(return_value={
        "id": 1,
        "status": "pending",
        "patient_id": 1,
        "file_url": "test.pdf"
    })
    service.get_analysis = AsyncMock(return_value={
        "id": 1,
        "status": "completed",
        "diagnosis": "Normal",
        "findings": {"heart_rate": 75}
    })
    service.list_analyses = AsyncMock(return_value={
        "items": [],
        "total": 0,
        "page": 1,
        "pages": 0
    })
    service.get_analyses_by_patient = AsyncMock(return_value=[])
    service.validate_analysis = AsyncMock(return_value=True)
    service.create_validation = AsyncMock(return_value={"id": 1, "status": "pending"})
    return service


@pytest.fixture
def mock_user_service():
    """Mock do serviço de usuário."""
    service = AsyncMock()
    service.authenticate = AsyncMock(return_value={
        "access_token": "test-token",
        "token_type": "bearer"
    })
    service.get_current_user = AsyncMock(return_value={
        "id": "user-123",
        "email": "test@example.com",
        "role": "user"
    })
    return service


@pytest.fixture
def mock_validation_service():
    """Mock do serviço de validação."""
    service = AsyncMock()
    service.validate_analysis = AsyncMock(return_value=True)
    service.create_validation = AsyncMock(return_value={
        "id": 1,
        "analysis_id": 1,
        "status": "pending"
    })
    return service


@pytest.fixture
def auth_headers():
    """Headers de autenticação para testes."""
    return {"Authorization": "Bearer test-token"}


@pytest.fixture
def test_user():
    """Dados de usuário para testes."""
    return {
        "id": "test-user-id",
        "email": "test@example.com",
        "name": "Test User",
        "role": "user",
        "is_active": True
    }


@pytest.fixture
def test_patient():
    """Dados de paciente para testes."""
    return {
        "id": 1,
        "name": "Paciente Teste",
        "cpf": "123.456.789-00",
        "birth_date": "1990-01-01",
        "gender": "M",
        "email": "paciente@example.com"
    }
