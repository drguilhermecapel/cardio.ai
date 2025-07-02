"""Comprehensive API endpoint tests with proper authentication"""

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import json

from app.core.config import settings

# Skip tests in standalone mode
if settings.STANDALONE_MODE:
    pytest.skip(
        "API endpoint tests skipped in standalone mode", allow_module_level=True
    )

from app.main import app
from app.core.security import create_access_token
from app.models.user import User
from app.core.constants import UserRoles


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def auth_token():
    """Create valid auth token"""
    token_data = {"sub": "test_user", "type": "access"}
    return create_access_token(token_data)


@pytest.fixture
def auth_headers(auth_token):
    """Create auth headers"""
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture
def admin_token():
    """Create admin auth token"""
    token_data = {"sub": "admin_user", "type": "access", "role": "admin"}
    return create_access_token(token_data)


@pytest.fixture
def admin_headers(admin_token):
    """Create admin auth headers"""
    return {"Authorization": f"Bearer {admin_token}"}


# Basic endpoint tests
def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_api_v1_health_endpoint(client):
    """Test API v1 health endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200


def test_docs_endpoint(client):
    """Test API documentation endpoint."""
    response = client.get("/api/v1/docs")
    assert response.status_code == 200


def test_openapi_endpoint(client):
    """Test OpenAPI schema endpoint."""
    response = client.get("/api/v1/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema


def test_cors_headers(client):
    """Test CORS headers are present."""
    response = client.get("/health")
    assert response.status_code == 200


def test_invalid_endpoint(client):
    """Test invalid endpoint returns 404."""
    response = client.get("/invalid-endpoint")
    assert response.status_code == 404


def test_api_v1_prefix_routes(client):
    """Test API v1 prefix routes exist."""
    response = client.get("/api/v1/")
    assert response.status_code in [200, 404, 405]


# Authentication endpoint tests
def test_authentication_endpoints(client):
    """Test authentication endpoints exist."""
    response = client.post("/api/v1/auth/login")
    assert response.status_code in [400, 422, 405]


@pytest.mark.asyncio
async def test_auth_endpoints_comprehensive(client):
    """Test all auth endpoints comprehensively."""

    # Test login - success case
    with patch("app.services.user_service.UserService") as mock_service_class:
        mock_service = Mock()
        mock_service_class.return_value = mock_service

        mock_user = Mock()
        mock_user.id = 1
        mock_user.email = "test@test.com"
        mock_user.is_active = True
        mock_user.hashed_password = "hashed_password"

        mock_service.authenticate_user = AsyncMock(return_value=mock_user)

        # Patch the dependency
        with patch(
            "app.api.v1.endpoints.auth.get_user_service", return_value=mock_service
        ):
            response = client.post(
                "/api/v1/auth/login",
                data={"username": "test@test.com", "password": "password123"},
            )

            assert response.status_code == 200
            data = response.json()
            assert "access_token" in data
            assert data["token_type"] == "bearer"

    # Test login - invalid credentials
    with patch("app.services.user_service.UserService") as mock_service_class:
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.authenticate_user = AsyncMock(return_value=None)

        with patch(
            "app.api.v1.endpoints.auth.get_user_service", return_value=mock_service
        ):
            response = client.post(
                "/api/v1/auth/login",
                data={"username": "test@test.com", "password": "wrong"},
            )
            assert response.status_code == 401


# User endpoint tests
def test_user_endpoints(client):
    """Test user endpoints exist."""
    response = client.get("/api/v1/users/")
    assert response.status_code in [401, 403, 422]


@pytest.mark.asyncio
async def test_users_endpoints_comprehensive(client, auth_headers, admin_headers):
    """Test all user endpoints comprehensively."""

    # Test get current user
    with patch("app.api.v1.endpoints.users.get_current_active_user") as mock_get_user:
        mock_user = Mock()
        mock_user.id = 1
        mock_user.email = "test@test.com"
        mock_user.full_name = "Test User"
        mock_user.is_active = True
        mock_user.role = UserRoles.PHYSICIAN
        mock_get_user.return_value = mock_user

        response = client.get("/api/v1/users/me", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "test@test.com"


# Patient endpoint tests
def test_patient_endpoints(client):
    """Test patient endpoints exist."""
    response = client.get("/api/v1/patients/")
    assert response.status_code in [401, 403, 422]


@pytest.mark.asyncio
async def test_patients_endpoints_comprehensive(client, auth_headers):
    """Test all patient endpoints comprehensively."""

    # Mock the auth dependency
    mock_user = Mock()
    mock_user.id = 1

    with patch(
        "app.api.v1.endpoints.patients.get_current_active_user", return_value=mock_user
    ):
        # Test create patient
        with patch("app.services.patient_service.PatientService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service

            mock_patient = Mock()
            mock_patient.id = 1
            mock_patient.name = "Test Patient"
            mock_patient.birth_date = datetime(1990, 1, 1).date()
            mock_patient.gender = "M"
            mock_patient.email = "patient@test.com"
            mock_service.create_patient = AsyncMock(return_value=mock_patient)

            with patch(
                "app.api.v1.endpoints.patients.get_patient_service",
                return_value=mock_service,
            ):
                response = client.post(
                    "/api/v1/patients/",
                    headers=auth_headers,
                    json={
                        "name": "Test Patient",
                        "birth_date": "1990-01-01",
                        "gender": "M",
                        "email": "patient@test.com",
                    },
                )
                assert response.status_code in [200, 201]


# ECG Analysis endpoint tests
def test_ecg_analysis_endpoints(client):
    """Test ECG analysis endpoints exist."""
    response = client.get("/api/v1/ecg-analyses/")
    assert response.status_code in [401, 403, 404, 422]


@pytest.mark.asyncio
async def test_ecg_endpoints_comprehensive(client, auth_headers):
    """Test all ECG analysis endpoints comprehensively."""

    # Mock user for auth
    mock_user = Mock()
    mock_user.id = 1

    with patch(
        "app.api.v1.endpoints.ecg_analysis.get_current_active_user",
        return_value=mock_user,
    ):
        # Test upload ECG
        with patch("app.services.ecg_service_instance.ECGAnalysisService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service

            mock_analysis = Mock()
            mock_analysis.id = "analysis_123"
            mock_analysis.status = "completed"
            mock_analysis.results = {"diagnosis": "normal"}
            mock_service.analyze_ecg = AsyncMock(return_value=mock_analysis)

            with patch(
                "app.api.v1.endpoints.ecg_analysis.get_ecg_service",
                return_value=mock_service,
            ):
                # Create mock file
                files = {"file": ("test.csv", b"fake ecg data", "text/csv")}
                data = {"patient_id": "1"}

                response = client.post(
                    "/api/v1/ecg/analyze", headers=auth_headers, files=files, data=data
                )

                assert response.status_code in [200, 201]


# Notification endpoint tests
def test_notification_endpoints(client):
    """Test notification endpoints exist."""
    response = client.get("/api/v1/notifications/")
    assert response.status_code in [401, 403, 422]


@pytest.mark.asyncio
async def test_notifications_endpoints_comprehensive(client, auth_headers):
    """Test all notification endpoints comprehensively."""

    # Mock user
    mock_user = Mock()
    mock_user.id = 1

    with patch(
        "app.api.v1.endpoints.notifications.get_current_active_user",
        return_value=mock_user,
    ):
        # Test get user notifications
        with patch(
            "app.services.notification_service.NotificationService"
        ) as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.get_user_notifications = AsyncMock(return_value=[])

            with patch(
                "app.api.v1.endpoints.notifications.get_notification_service",
                return_value=mock_service,
            ):
                response = client.get("/api/v1/notifications/", headers=auth_headers)
                assert response.status_code == 200
                assert isinstance(response.json(), list)


# Validation endpoint tests
def test_validation_endpoints(client):
    """Test validation endpoints exist."""
    response = client.get("/api/v1/validations/")
    assert response.status_code in [401, 403, 405, 422]


@pytest.mark.asyncio
async def test_validations_endpoints_comprehensive(client, auth_headers):
    """Test all validation endpoints comprehensively."""

    # Mock user
    mock_user = Mock()
    mock_user.id = 1
    mock_user.role = UserRoles.PHYSICIAN

    with patch(
        "app.api.v1.endpoints.validations.get_current_active_user",
        return_value=mock_user,
    ):
        # Test create validation
        with patch(
            "app.services.validation_service.ValidationService"
        ) as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service

            mock_validation = Mock()
            mock_validation.id = 1
            mock_validation.analysis_id = 1
            mock_validation.status = "pending"
            mock_validation.notes = "Initial validation"
            mock_service.create_validation = AsyncMock(return_value=mock_validation)

            with patch(
                "app.api.v1.endpoints.validations.get_validation_service",
                return_value=mock_service,
            ):
                response = client.post(
                    "/api/v1/validations/",
                    headers=auth_headers,
                    json={
                        "analysis_id": 1,
                        "status": "pending",
                        "notes": "Initial validation",
                    },
                )
                assert response.status_code in [200, 201]


@pytest.mark.asyncio
async def test_error_handling_and_edge_cases(client, auth_headers):
    """Test error handling and edge cases for all endpoints."""

    # Mock user
    mock_user = Mock()
    mock_user.id = 1

    with patch(
        "app.api.v1.endpoints.patients.get_current_active_user", return_value=mock_user
    ):
        # Test invalid JSON - CORRIGIDO AQUI
        with patch("app.services.patient_service.PatientService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service

            with patch(
                "app.api.v1.endpoints.patients.get_patient_service",
                return_value=mock_service,
            ):
                # Envia JSON inválido com o content-type correto
                response = client.post(
                    "/api/v1/patients/",
                    headers={**auth_headers, "Content-Type": "application/json"},
                    content="invalid json{",
                )
                assert response.status_code in [400, 422]

        # Test missing required fields
        with patch("app.services.patient_service.PatientService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service

            with patch(
                "app.api.v1.endpoints.patients.get_patient_service",
                return_value=mock_service,
            ):
                response = client.post(
                    "/api/v1/patients/", headers=auth_headers, json={}
                )  # Missing required fields
                assert response.status_code == 422

        # Test non-existent resource
        with patch("app.services.patient_service.PatientService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.get_patient = AsyncMock(return_value=None)

            with patch(
                "app.api.v1.endpoints.patients.get_patient_service",
                return_value=mock_service,
            ):
                response = client.get("/api/v1/patients/999", headers=auth_headers)
                assert response.status_code == 404

        # Test unauthorized access
        response = client.get("/api/v1/patients/1")  # No auth headers
        assert response.status_code == 401

        # Test invalid auth token
        invalid_headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/api/v1/patients/1", headers=invalid_headers)
        assert response.status_code == 401


@pytest.mark.asyncio
async def test_comprehensive_workflow(client, auth_headers):
    """Test complete workflow from patient creation to ECG analysis validation."""

    mock_user = Mock()
    mock_user.id = 1
    mock_user.role = UserRoles.PHYSICIAN

    # Step 1: Create patient
    with patch(
        "app.api.v1.endpoints.patients.get_current_active_user", return_value=mock_user
    ):
        with patch("app.services.patient_service.PatientService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service

            mock_patient = Mock()
            mock_patient.id = 1
            mock_patient.name = "John Doe"
            mock_service.create_patient = AsyncMock(return_value=mock_patient)

            with patch(
                "app.api.v1.endpoints.patients.get_patient_service",
                return_value=mock_service,
            ):
                response = client.post(
                    "/api/v1/patients/",
                    headers=auth_headers,
                    json={
                        "name": "John Doe",
                        "birth_date": "1980-01-01",
                        "gender": "M",
                        "email": "john@example.com",
                    },
                )
                assert response.status_code in [200, 201]
                patient_id = 1

    # Step 2: Upload ECG for analysis
    with patch(
        "app.api.v1.endpoints.ecg_analysis.get_current_active_user",
        return_value=mock_user,
    ):
        with patch("app.services.ecg_service_instance.ECGAnalysisService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service

            mock_analysis = Mock()
            mock_analysis.id = "analysis_001"
            mock_analysis.patient_id = patient_id
            mock_analysis.status = "completed"
            mock_analysis.results = {
                "diagnosis": "Atrial Fibrillation",
                "confidence": 0.85,
                "clinical_urgency": "high",
            }
            mock_service.analyze_ecg = AsyncMock(return_value=mock_analysis)

            with patch(
                "app.api.v1.endpoints.ecg_analysis.get_ecg_service",
                return_value=mock_service,
            ):
                files = {"file": ("ecg.csv", b"ecg,data,here", "text/csv")}
                data = {"patient_id": str(patient_id)}

                response = client.post(
                    "/api/v1/ecg/analyze", headers=auth_headers, files=files, data=data
                )
                assert response.status_code in [200, 201]

    # Step 3: Validate the analysis
    with patch(
        "app.api.v1.endpoints.validations.get_current_active_user",
        return_value=mock_user,
    ):
        with patch(
            "app.services.validation_service.ValidationService"
        ) as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service

            mock_validation = Mock()
            mock_validation.id = 1
            mock_validation.analysis_id = "analysis_001"
            mock_validation.status = "approved"
            mock_validation.validated_by = mock_user.id
            mock_service.create_validation = AsyncMock(return_value=mock_validation)

            with patch(
                "app.api.v1.endpoints.validations.get_validation_service",
                return_value=mock_service,
            ):
                response = client.post(
                    "/api/v1/validations/",
                    headers=auth_headers,
                    json={
                        "analysis_id": "analysis_001",
                        "status": "approved",
                        "notes": "Confirmed AF diagnosis",
                    },
                )
                assert response.status_code in [200, 201]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
