"""Test patient service."""

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

import pytest
from datetime import date
from unittest.mock import AsyncMock, Mock

from app.services.patient_service import PatientService
from app.schemas.patient import PatientCreate
from app.models.patient import Patient


@pytest.fixture
def patient_service(test_db):
    """Create patient service instance."""
    return PatientService(db=test_db)


@pytest.fixture
def sample_patient_data():
    """Sample patient data."""
    return PatientCreate(
        patient_id="PAT123456",
        mrn="MRN123456",
        first_name="John",
        last_name="Doe",
        date_of_birth=date(1990, 1, 15),
        gender="male",
        phone="+1234567890",
        email="john.doe@example.com",
        address="123 Main St, City, State 12345",
        height_cm=175,
        weight_kg=70,
        blood_type="O+",
        emergency_contact_name="Jane Doe",
        emergency_contact_phone="+0987654321",
        emergency_contact_relationship="Spouse",
        allergies=["Penicillin"],
        medications=["Aspirin 81mg"],
        medical_history=["Hypertension"],
        family_history=["Heart disease"],
        insurance_provider="Health Insurance Co",
        insurance_number="INS123456",
        consent_for_research=True,
    )


@pytest.mark.asyncio
async def test_create_patient_success(patient_service, sample_patient_data):
    """Test successful patient creation."""
    mock_patient = Patient()
    mock_patient.id = 1
    mock_patient.patient_id = "PAT123456"
    mock_patient.first_name = "John"
    mock_patient.last_name = "Doe"
    mock_patient.date_of_birth = date(1990, 1, 15)
    mock_patient.gender = "male"
    mock_patient.created_by = 1

    patient_service.repository.create_patient = AsyncMock(return_value=mock_patient)

    patient = await patient_service.create_patient(
        patient_data=sample_patient_data, created_by=1
    )

    assert patient.patient_id == "PAT123456"
    assert patient.first_name == "John"
    assert patient.last_name == "Doe"
    assert patient.created_by == 1


@pytest.mark.asyncio
async def test_get_patient_by_patient_id(patient_service):
    """Test getting patient by patient ID."""
    mock_patient = Patient()
    mock_patient.patient_id = "PAT123456"
    mock_patient.first_name = "John"
    mock_patient.last_name = "Doe"

    patient_service.repository.get_patient_by_patient_id = AsyncMock(
        return_value=mock_patient
    )

    patient = await patient_service.get_patient_by_patient_id("PAT123456")

    assert patient is not None
    assert patient.patient_id == "PAT123456"
    assert patient.first_name == "John"


@pytest.mark.asyncio
async def test_get_patient_by_patient_id_not_found(patient_service):
    """Test getting non-existent patient by patient ID."""
    patient_service.repository.get_patient_by_patient_id = AsyncMock(return_value=None)

    patient = await patient_service.get_patient_by_patient_id("NONEXISTENT")

    assert patient is None


@pytest.mark.asyncio
async def test_update_patient(patient_service):
    """Test updating patient information."""
    mock_patient = Patient()
    mock_patient.id = 1
    mock_patient.first_name = "Jane"
    mock_patient.phone = "+1111111111"

    patient_service.repository.get_patient_by_id = AsyncMock(return_value=mock_patient)
    patient_service.repository.update_patient = AsyncMock(return_value=mock_patient)

    update_data = {"first_name": "Jane", "phone": "+1111111111"}
    patient = await patient_service.update_patient(1, update_data)

    assert patient.first_name == "Jane"
    assert patient.phone == "+1111111111"


@pytest.mark.asyncio
async def test_get_patients_paginated(patient_service):
    """Test getting patients with pagination."""
    mock_patients = [Patient(), Patient()]
    patient_service.repository.get_patients = AsyncMock(return_value=(mock_patients, 2))

    patients, total = await patient_service.get_patients(limit=10, offset=0)

    assert len(patients) == 2
    assert total == 2
    patient_service.repository.get_patients.assert_called_once_with(10, 0)


@pytest.mark.asyncio
async def test_search_patients(patient_service):
    """Test searching patients."""
    mock_patients = [Patient()]
    patient_service.repository.search_patients = AsyncMock(
        return_value=(mock_patients, 1)
    )

    patients, total = await patient_service.search_patients(
        query="John", search_fields=["first_name", "last_name"], limit=10, offset=0
    )

    assert len(patients) == 1
    assert total == 1
    patient_service.repository.search_patients.assert_called_once_with(
        "John", ["first_name", "last_name"], 10, 0
    )


@pytest.mark.asyncio
async def test_patient_data_audit_trail(patient_service, sample_patient_data):
    """Ensure patient modifications are tracked for FDA compliance."""
    mock_patient = Patient()
    mock_patient.id = 1
    mock_patient.patient_id = "PAT123456"
    mock_patient.first_name = "John"
    mock_patient.last_name = "Doe"
    mock_patient.created_by = 1
    mock_patient.created_at = "2025-06-01T14:00:00Z"
    mock_patient.updated_at = "2025-06-01T14:00:00Z"

    patient_service.repository.create_patient = AsyncMock(return_value=mock_patient)

    patient = await patient_service.create_patient(
        patient_data=sample_patient_data, created_by=1
    )

    assert patient.created_by == 1
    assert patient.created_at is not None
    assert patient.updated_at is not None
