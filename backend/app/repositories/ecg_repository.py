"""
ECG Analysis Repository - Data access layer for ECG analyses.
"""

from typing import Any

from sqlalchemy import and_, desc, func, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from app.core.constants import AnalysisStatus
from app.models.ecg_analysis import ECGAnalysis


class ECGRepository:
    """Repository for ECG analysis data access."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create_analysis(self, analysis: ECGAnalysis) -> ECGAnalysis:
        """Create a new ECG analysis."""
        self.db.add(analysis)
        await self.db.commit()
        await self.db.refresh(analysis)
        return analysis

    async def get_analysis_by_id(self, analysis_id: int) -> ECGAnalysis | None:
        """Get analysis by ID."""
        stmt = (
            select(ECGAnalysis)
            .options(
                selectinload(ECGAnalysis.patient),
                selectinload(ECGAnalysis.created_by_user),
                selectinload(ECGAnalysis.measurements),
                selectinload(ECGAnalysis.validations),
            )
            .where(ECGAnalysis.id == analysis_id)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_analysis_by_analysis_id(self, analysis_id: str) -> ECGAnalysis | None:
        """Get analysis by analysis ID string."""
        stmt = (
            select(ECGAnalysis)
            .options(
                selectinload(ECGAnalysis.patient),
                selectinload(ECGAnalysis.created_by_user),
                selectinload(ECGAnalysis.measurements),
                selectinload(ECGAnalysis.validations),
            )
            .where(ECGAnalysis.analysis_id == analysis_id)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_analyses_by_patient(
        self, patient_id: int, limit: int = 50, offset: int = 0
    ) -> list[ECGAnalysis]:
        """Get analyses by patient ID."""
        stmt = (
            select(ECGAnalysis)
            .options(
                selectinload(ECGAnalysis.patient),
                selectinload(ECGAnalysis.created_by_user),
            )
            .where(ECGAnalysis.patient_id == patient_id)
            .order_by(desc(ECGAnalysis.created_at))
            .limit(limit)
            .offset(offset)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def search_analyses(
        self,
        filters: dict[str, Any],
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[ECGAnalysis], int]:
        """Search analyses with filters."""
        stmt = select(ECGAnalysis).options(
            selectinload(ECGAnalysis.patient),
            selectinload(ECGAnalysis.created_by_user),
        )

        conditions = []

        if "patient_id" in filters and filters["patient_id"]:
            conditions.append(ECGAnalysis.patient_id == filters["patient_id"])

        if "status" in filters and filters["status"]:
            conditions.append(ECGAnalysis.status == filters["status"])

        if "clinical_urgency" in filters and filters["clinical_urgency"]:
            conditions.append(
                ECGAnalysis.clinical_urgency == filters["clinical_urgency"]
            )

        if "diagnosis_category" in filters and filters["diagnosis_category"]:
            conditions.append(
                ECGAnalysis.diagnosis_category == filters["diagnosis_category"]
            )

        if "date_from" in filters and filters["date_from"]:
            conditions.append(ECGAnalysis.created_at >= filters["date_from"])

        if "date_to" in filters and filters["date_to"]:
            conditions.append(ECGAnalysis.created_at <= filters["date_to"])

        if "is_validated" in filters and filters["is_validated"] is not None:
            conditions.append(ECGAnalysis.is_validated == filters["is_validated"])

        if (
            "requires_validation" in filters
            and filters["requires_validation"] is not None
        ):
            conditions.append(
                ECGAnalysis.validation_required == filters["requires_validation"]
            )

        if "created_by" in filters and filters["created_by"]:
            conditions.append(ECGAnalysis.created_by == filters["created_by"])

        if conditions:
            stmt = stmt.where(and_(*conditions))

        count_stmt = select(func.count(ECGAnalysis.id))
        if conditions:
            count_stmt = count_stmt.where(and_(*conditions))

        count_result = await self.db.execute(count_stmt)
        total = count_result.scalar()

        stmt = stmt.order_by(desc(ECGAnalysis.created_at)).limit(limit).offset(offset)

        result = await self.db.execute(stmt)
        analyses = list(result.scalars().all())

        return analyses, total or 0

    async def update_analysis(
        self, analysis_id: int, update_data: dict[str, Any]
    ) -> ECGAnalysis | None:
        """Update analysis."""
        stmt = select(ECGAnalysis).where(ECGAnalysis.id == analysis_id)
        result = await self.db.execute(stmt)
        analysis = result.scalar_one_or_none()

        if analysis:
            for key, value in update_data.items():
                if hasattr(analysis, key):
                    setattr(analysis, key, value)

            await self.db.commit()
            await self.db.refresh(analysis)

        return analysis

    async def update_analysis_status(
        self, analysis_id: int, status: AnalysisStatus
    ) -> bool:
        """Update analysis status."""
        stmt = select(ECGAnalysis).where(ECGAnalysis.id == analysis_id)
        result = await self.db.execute(stmt)
        analysis = result.scalar_one_or_none()

        if analysis:
            analysis.status = status
            await self.db.commit()
            return True

        return False

    async def delete_analysis(self, analysis_id: int) -> bool:
        """Delete analysis (soft delete)."""
        return (
            await self.update_analysis(
                analysis_id, {"is_active": False, "deleted_at": func.now()}
            )
            is not None
        )

    async def create_measurement(self, measurement: dict) -> dict:
        """Create ECG measurement."""
        self.db.add(measurement)
        await self.db.commit()
        await self.db.refresh(measurement)
        return measurement

    async def get_measurements_by_analysis(
        self, analysis_id: int
    ) -> list[dict]:
        """Get measurements by analysis ID."""
        stmt = (
            select(dict)
            .where(dict.analysis_id == analysis_id)
            .order_by(dict.measurement_type, dict.lead_name)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def create_annotation(self, annotation: dict) -> dict:
        """Create ECG annotation."""
        self.db.add(annotation)
        await self.db.commit()
        await self.db.refresh(annotation)
        return annotation

    async def get_annotations_by_analysis(
        self, analysis_id: int
    ) -> list[dict]:
        """Get annotations by analysis ID."""
        stmt = (
            select(dict)
            .where(dict.analysis_id == analysis_id)
            .order_by(dict.time_ms)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_critical_analyses(self, limit: int = 20) -> list[ECGAnalysis]:
        """Get analyses requiring immediate attention."""
        stmt = (
            select(ECGAnalysis)
            .options(
                selectinload(ECGAnalysis.patient),
                selectinload(ECGAnalysis.created_by_user),
            )
            .where(
                or_(
                    ECGAnalysis.requires_immediate_attention.is_(True),
                    ECGAnalysis.clinical_urgency == "critical",
                )
            )
            .where(ECGAnalysis.status == AnalysisStatus.COMPLETED)
            .where(ECGAnalysis.is_validated.is_(False))
            .order_by(desc(ECGAnalysis.created_at))
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_pending_validations(self, limit: int = 50) -> list[ECGAnalysis]:
        """Get analyses pending validation."""
        stmt = (
            select(ECGAnalysis)
            .options(
                selectinload(ECGAnalysis.patient),
                selectinload(ECGAnalysis.created_by_user),
            )
            .where(ECGAnalysis.status == AnalysisStatus.COMPLETED)
            .where(ECGAnalysis.validation_required.is_(True))
            .where(ECGAnalysis.is_validated.is_(False))
            .order_by(desc(ECGAnalysis.clinical_urgency), desc(ECGAnalysis.created_at))
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_analysis_statistics(
        self, date_from: str | None = None, date_to: str | None = None
    ) -> dict[str, Any]:
        """Get analysis statistics."""
        base_query = select(ECGAnalysis)

        if date_from:
            base_query = base_query.where(ECGAnalysis.created_at >= date_from)
        if date_to:
            base_query = base_query.where(ECGAnalysis.created_at <= date_to)

        total_stmt = select(func.count(ECGAnalysis.id))
        if date_from:
            total_stmt = total_stmt.where(ECGAnalysis.created_at >= date_from)
        if date_to:
            total_stmt = total_stmt.where(ECGAnalysis.created_at <= date_to)

        total_result = await self.db.execute(total_stmt)
        total_analyses = total_result.scalar()

        status_stmt = select(ECGAnalysis.status, func.count(ECGAnalysis.id)).group_by(
            ECGAnalysis.status
        )
        if date_from:
            status_stmt = status_stmt.where(ECGAnalysis.created_at >= date_from)
        if date_to:
            status_stmt = status_stmt.where(ECGAnalysis.created_at <= date_to)

        status_result = await self.db.execute(status_stmt)
        status_counts: dict[str, int] = {
            str(status): count for status, count in status_result.all()
        }

        critical_stmt = select(func.count(ECGAnalysis.id)).where(
            ECGAnalysis.requires_immediate_attention.is_(True)
        )
        if date_from:
            critical_stmt = critical_stmt.where(ECGAnalysis.created_at >= date_from)
        if date_to:
            critical_stmt = critical_stmt.where(ECGAnalysis.created_at <= date_to)

        critical_result = await self.db.execute(critical_stmt)
        critical_count = critical_result.scalar()

        return {
            "total_analyses": total_analyses,
            "status_distribution": status_counts,
            "critical_analyses": critical_count,
            "validation_rate": (
                status_counts.get("completed", 0)
                - ((total_analyses or 0) - sum(status_counts.values()))
            )
            / max(total_analyses or 1, 1)
            * 100,
        }
