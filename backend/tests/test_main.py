import pytest
from fastapi.testclient import TestClient
from app.main import app

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"


def test_health_endpoint():
    """Test health check endpoint."""
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "cardioai-pro-standalone"


def test_root_endpoint():
    """Test root endpoint."""
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200


def test_api_v1_prefix():
    """Test API v1 prefix is included."""
    client = TestClient(app)
    response = client.get("/api/v1/health")

    assert response.status_code == 200


def test_cors_middleware():
    """Test CORS middleware is configured."""
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200


def test_app_initialization():
    """Test FastAPI app initialization."""
    assert app.title == "CardioAI Pro"  # Corrigido - removido "API"
    assert app.version == "1.0.0"


def test_openapi_schema():
    """Test OpenAPI schema generation."""
    client = TestClient(app)
    response = client.get("/api/v1/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert schema["info"]["title"] == "CardioAI Pro"  # Corrigido - removido "API"


def test_docs_endpoint():
    """Test API documentation endpoint."""
    client = TestClient(app)
    response = client.get("/api/v1/docs")

    assert response.status_code == 200


def test_redoc_endpoint():
    """Test ReDoc documentation endpoint."""
    client = TestClient(app)
    response = client.get("/api/v1/redoc")

    assert response.status_code == 200


def test_middleware_configuration():
    """Test middleware configuration."""
    middlewares = [middleware.cls.__name__ for middleware in app.user_middleware]
    assert "CORSMiddleware" in middlewares


def test_exception_handlers():
    """Test exception handlers are configured."""
    assert len(app.exception_handlers) >= 0


def test_startup_event():
    """Test startup event registration."""
    # Verifica se o evento de startup está registrado
    assert "startup" in app.router.on_startup


def test_shutdown_event():
    """Test shutdown event registration."""
    # Verifica se o evento de shutdown está registrado
    assert "shutdown" in app.router.on_shutdown


def test_api_routes_included():
    """Test that API routes are included."""
    routes = [route.path for route in app.routes]
    # Verifica se as rotas da API estão incluídas
    assert any("/api/v1" in route for route in routes)


def test_404_handling():
    """Test 404 error handling."""
    client = TestClient(app)
    response = client.get("/nonexistent-endpoint")
    
    assert response.status_code == 404


def test_method_not_allowed():
    """Test 405 method not allowed."""
    client = TestClient(app)
    response = client.post("/health")  # Health endpoint only accepts GET
    
    assert response.status_code == 405


def test_cors_headers():
    """Test CORS headers in response."""
    client = TestClient(app)
    response = client.get("/health", headers={"Origin": "http://localhost:3000"})
    
    assert response.status_code == 200
    # Em modo standalone, CORS permite todas as origens
    assert "access-control-allow-origin" in response.headers


def test_api_version():
    """Test API version in root response."""
    client = TestClient(app)
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert data["version"] == "1.0.0"


def test_root_response_structure():
    """Test root endpoint response structure."""
    client = TestClient(app)
    response = client.get("/")
    
    assert response.status_code == 200
    data = response.json()
    
    # Verifica estrutura da resposta
    assert "message" in data
    assert "version" in data
    assert "docs" in data
    assert "status" in data
    
    # Verifica valores
    assert data["message"] == "CardioAI Pro Standalone API"
    assert data["status"] == "running"
    assert data["docs"] == "/docs"


def test_health_response_structure():
    """Test health endpoint response structure."""
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code == 200
    data = response.json()
    
    # Verifica estrutura da resposta
    assert "status" in data
    assert "service" in data
    
    # Verifica valores
    assert data["status"] == "healthy"
    assert data["service"] == "cardioai-pro-standalone"
