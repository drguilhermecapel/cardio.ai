import pytest
from unittest.mock import Mock, patch, AsyncMock
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.patient_repository import PatientRepository
from app.models.patient import Patient
from app.schemas.patient import PatientCreate, PatientUpdate
from datetime import date

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"


class TestPatientRepositoryComprehensive:
    """Comprehensive test coverage for PatientRepository to reach 80% coverage"""

    @pytest.fixture
    def mock_db(self):
        return Mock(spec=AsyncSession)

    @pytest.fixture
    def repository(self, mock_db):
        return PatientRepository(mock_db)

    @pytest.fixture
    def sample_patient_data(self):
        return {
            "patient_id": "PAT123456",
            "first_name": "John",
            "last_name": "Doe",
            "date_of_birth": date(1990, 1, 1),
            "gender": "male",
            "email": "john.doe@example.com",
            "phone": "+1234567890",
            "mrn": "MRN123456",
        }

    @pytest.fixture
    def sample_patient(self, sample_patient_data):
        patient = Patient()
        for key, value in sample_patient_data.items():
            setattr(patient, key, value)
        patient.id = 1
        patient.is_active = True
        patient.created_by = 1
        return patient

    def test_repository_initialization(self, mock_db):
        """Test repository initialization"""
        repo = PatientRepository(mock_db)
        assert repo.db == mock_db

    @pytest.mark.asyncio
    async def test_create_patient_success(self, repository, mock_db, sample_patient):
        """Test successful patient creation"""
        mock_db.add.return_value = None
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock()

        result = await repository.create_patient(sample_patient)

        mock_db.add.assert_called_once_with(sample_patient)
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once_with(sample_patient)
        assert result == sample_patient

    @pytest.mark.asyncio
    async def test_get_patient_by_id_success(self, repository, mock_db, sample_patient):
        """Test successful patient retrieval by ID"""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_patient
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_patient_by_id(1)

        assert result == sample_patient
        mock_db.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_patient_by_id_not_found(self, repository, mock_db):
        """Test patient retrieval when not found"""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = None
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_patient_by_id(999)

        assert result is None

    @pytest.mark.asyncio
    async def test_get_patient_by_patient_id_success(
        self, repository, mock_db, sample_patient
    ):
        """Test successful patient retrieval by patient ID"""
        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_patient
        mock_db.execute = AsyncMock(return_value=mock_result)

        result = await repository.get_patient_by_patient_id("PAT123456")

        assert result == sample_patient

    @pytest.mark.asyncio
    async def test_get_patients_success(self, repository, mock_db, sample_patient):
        """Test successful retrieval of patients with pagination"""
        mock_patients = [sample_patient]

        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 1

        mock_patients_result = Mock()
        mock_patients_result.scalars.return_value.all.return_value = mock_patients

        mock_db.execute = AsyncMock(
            side_effect=[mock_count_result, mock_patients_result]
        )

        result = await repository.get_patients(limit=50, offset=0)

        assert result == (mock_patients, 1)
        assert mock_db.execute.call_count == 2

    @pytest.mark.asyncio
    async def test_update_patient_success(self, repository, mock_db, sample_patient):
        """Test successful patient update"""
        update_data = {"first_name": "Jane", "email": "jane.doe@example.com"}

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = sample_patient
        mock_db.execute = AsyncMock(return_value=mock_result)
        mock_db.commit = AsyncMock()
        mock_db.refresh = AsyncMock()

        result = await repository.update_patient(1, update_data)

        assert sample_patient.first_name == "Jane"
        assert sample_patient.email == "jane.doe@example.com"
        mock_db.commit.assert_called_once()
        mock_db.refresh.assert_called_once()
        assert result == sample_patient

    @pytest.mark.asyncio
    async def test_search_patients_success(self, repository, mock_db, sample_patient):
        """Test patient search functionality"""
        mock_patients = [sample_patient]

        mock_count_result = Mock()
        mock_count_result.scalar.return_value = 1

        mock_search_result = Mock()
        mock_search_result.scalars.return_value.all.return_value = mock_patients

        mock_db.execute = AsyncMock(side_effect=[mock_count_result, mock_search_result])

        result = await repository.search_patients("John", ["first_name", "last_name"])

        assert result == (mock_patients, 1)
        assert mock_db.execute.call_count == 2
