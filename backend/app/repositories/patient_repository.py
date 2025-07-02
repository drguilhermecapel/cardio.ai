"""
Patient Repository - Data access layer for patients.
"""

from typing import Any

from sqlalchemy import and_, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.models.patient import Patient


class PatientRepository:
    """Repository for patient data access."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create_patient(self, patient: Patient) -> Patient:
        """Create a new patient."""
        self.db.add(patient)
        await self.db.commit()
        await self.db.refresh(patient)
        return patient

    async def get_patient_by_id(self, patient_id: int) -> Patient | None:
        """Get patient by ID."""
        stmt = select(Patient).where(Patient.id == patient_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_patient_by_patient_id(self, patient_id: str) -> Patient | None:
        """Get patient by patient ID string."""
        stmt = select(Patient).where(Patient.patient_id == patient_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def update_patient(
        self, patient_id: int, update_data: dict[str, Any]
    ) -> Patient | None:
        """Update patient."""
        stmt = select(Patient).where(Patient.id == patient_id)
        result = await self.db.execute(stmt)
        patient = result.scalar_one_or_none()

        if patient:
            for key, value in update_data.items():
                if hasattr(patient, key):
                    setattr(patient, key, value)

            await self.db.commit()
            await self.db.refresh(patient)

        return patient

    async def get_patients(
        self, limit: int = 50, offset: int = 0
    ) -> tuple[list[Patient], int]:
        """Get patients with pagination."""
        count_stmt = select(func.count(Patient.id)).where(Patient.is_active.is_(True))
        count_result = await self.db.execute(count_stmt)
        total = count_result.scalar()

        stmt = (
            select(Patient)
            .where(Patient.is_active.is_(True))
            .order_by(Patient.last_name, Patient.first_name)
            .limit(limit)
            .offset(offset)
        )
        result = await self.db.execute(stmt)
        patients = list(result.scalars().all())

        return patients, total or 0

    async def search_patients(
        self, query: str, search_fields: list[str], limit: int = 50, offset: int = 0
    ) -> tuple[list[Patient], int]:
        """Search patients."""
        conditions = []
        query_lower = query.lower()

        for field in search_fields:
            if hasattr(Patient, field):
                field_attr = getattr(Patient, field)
                conditions.append(func.lower(field_attr).contains(query_lower))

        if not conditions:
            return [], 0

        count_stmt = select(func.count(Patient.id)).where(
            and_(Patient.is_active.is_(True), or_(*conditions))
        )
        count_result = await self.db.execute(count_stmt)
        total = count_result.scalar()

        stmt = (
            select(Patient)
            .where(and_(Patient.is_active.is_(True), or_(*conditions)))
            .order_by(Patient.last_name, Patient.first_name)
            .limit(limit)
            .offset(offset)
        )
        result = await self.db.execute(stmt)
        patients = list(result.scalars().all())

        return patients, total or 0
