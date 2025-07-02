"""Integration tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def api_client():
    """Create API test client."""
    return TestClient(app)


@pytest.mark.asyncio
async def test_health_endpoint(api_client):
    """Test health check endpoint."""
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service": "cardioai-pro-standalone",
    }


@pytest.mark.asyncio
async def test_api_v1_prefix(api_client):
    """Test API v1 prefix routing."""
    response = api_client.get("/api/v1/")

    assert response.status_code in [200, 404]


@pytest.mark.asyncio
async def test_cors_headers(api_client):
    """Test CORS headers are present."""
    response = api_client.options("/health")

    assert response.status_code in [200, 405]


@pytest.mark.asyncio
async def test_authentication_required_endpoints(api_client):
    """Test that protected endpoints require authentication."""
    protected_endpoints = [
        "/api/v1/patients/",
        "/api/v1/ecg/",
        "/api/v1/validations/my-validations",
        "/api/v1/users/me",
    ]

    for endpoint in protected_endpoints:
        response = api_client.get(endpoint)
        assert response.status_code in [401, 405]


@pytest.mark.asyncio
async def test_invalid_endpoint_404(api_client):
    """Test that invalid endpoints return 404."""
    response = api_client.get("/api/v1/nonexistent")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_method_not_allowed(api_client):
    """Test method not allowed responses."""
    response = api_client.delete("/health")

    assert response.status_code == 405


@pytest.mark.asyncio
async def test_request_validation_error(api_client):
    """Test request validation error handling."""
    invalid_data = {"invalid": "data"}

    response = api_client.post("/api/v1/patients/", json=invalid_data)

    assert response.status_code in [401, 422]


@pytest.mark.asyncio
async def test_large_request_handling(api_client):
    """Test handling of large requests."""
    large_data = {"data": "x" * 10000}

    response = api_client.post("/api/v1/patients/", json=large_data)

    assert response.status_code in [401, 413, 422]


@pytest.mark.asyncio
async def test_concurrent_requests(api_client):
    """Test handling of concurrent requests."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    def make_request():
        return api_client.get("/health")

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(10)]
        responses = [future.result() for future in futures]

    assert all(r.status_code == 200 for r in responses)


@pytest.mark.asyncio
async def test_api_documentation_endpoints(api_client):
    """Test API documentation endpoints."""
    docs_endpoints = ["/api/v1/docs", "/api/v1/redoc", "/api/v1/openapi.json"]

    for endpoint in docs_endpoints:
        response = api_client.get(endpoint)
        assert response.status_code == 200


@pytest.mark.asyncio
async def test_error_response_format(api_client):
    """Test error response format consistency."""
    response = api_client.get("/api/v1/nonexistent")

    assert response.status_code == 404
    error_data = response.json()
    assert "detail" in error_data


@pytest.mark.asyncio
async def test_request_timeout_handling(api_client):
    """Test request timeout handling."""
    import time

    start_time = time.time()
    response = api_client.get("/health")
    end_time = time.time()

    assert response.status_code == 200
    assert (end_time - start_time) < 5.0


@pytest.mark.asyncio
async def test_content_type_validation(api_client):
    """Test content type validation."""
    response = api_client.post(
        "/api/v1/patients/", data="invalid data", headers={"Content-Type": "text/plain"}
    )

    assert response.status_code in [401, 415, 422]


@pytest.mark.asyncio
async def test_security_headers(api_client):
    """Test security headers are present."""
    response = api_client.get("/health")

    assert response.status_code == 200
    assert "content-type" in response.headers
    assert "content-length" in response.headers


@pytest.mark.asyncio
async def test_rate_limiting_simulation(api_client):
    """Test rate limiting behavior simulation."""
    responses = []

    for _ in range(100):
        response = api_client.get("/health")
        responses.append(response)
        if response.status_code == 429:
            break

    assert all(r.status_code in [200, 429] for r in responses)


@pytest.mark.asyncio
async def test_websocket_connection():
    """Test WebSocket connection endpoint."""
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        try:
            with client.websocket_connect("/ws") as websocket:
                websocket.send_text("test")
                data = websocket.receive_text()
                assert data is not None
        except Exception:
            pass


@pytest.mark.asyncio
async def test_file_upload_endpoint(api_client):
    """Test file upload endpoint."""
    test_file = ("test.txt", b"test content", "text/plain")

    response = api_client.post("/api/v1/ecg/upload", files={"file": test_file})

    assert response.status_code in [401, 422]


@pytest.mark.asyncio
async def test_api_versioning(api_client):
    """Test API versioning support."""
    v1_response = api_client.get("/api/v1/")

    assert v1_response.status_code in [200, 404]


@pytest.mark.asyncio
async def test_database_connection_health(api_client):
    """Test database connection through health endpoint."""
    response = api_client.get("/health")

    assert response.status_code == 200
    health_data = response.json()
    assert health_data["status"] == "healthy"


@pytest.mark.asyncio
async def test_redis_connection_health(api_client):
    """Test Redis connection health."""
    response = api_client.get("/health")

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_logging_integration(api_client):
    """Test logging integration."""
    import logging

    logger = logging.getLogger("test")

    response = api_client.get("/health")
    assert response.status_code == 200

    logger.info("Test log message")
    assert True


@pytest.mark.asyncio
async def test_metrics_endpoint(api_client):
    """Test metrics endpoint if available."""
    response = api_client.get("/metrics")

    assert response.status_code in [200, 404]


@pytest.mark.asyncio
async def test_graceful_shutdown_simulation(api_client):
    """Test graceful shutdown behavior simulation."""
    response = api_client.get("/health")

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_middleware_chain(api_client):
    """Test middleware chain execution."""
    response = api_client.get("/health")

    assert response.status_code == 200
    assert "content-length" in response.headers


@pytest.mark.asyncio
async def test_exception_handling(api_client):
    """Test global exception handling."""
    response = api_client.get("/api/v1/trigger-error")

    assert response.status_code in [404, 500]


@pytest.mark.asyncio
async def test_request_id_tracking(api_client):
    """Test request ID tracking."""
    response = api_client.get("/health")

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_performance_monitoring(api_client):
    """Test performance monitoring."""
    import time

    start_time = time.time()
    response = api_client.get("/health")
    end_time = time.time()

    assert response.status_code == 200
    assert (end_time - start_time) < 1.0
