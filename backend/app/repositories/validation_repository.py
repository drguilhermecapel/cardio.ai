"""
Validation Repository - Data access layer for validations.
"""

from typing import Any

from sqlalchemy import and_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from app.core.constants import UserRoles, ValidationStatus
from app.models.ecg_analysis import ECGAnalysis
from app.models.user import User
from app.models.validation import (
    QualityMetric,
    Validation,
    ValidationResult,
    ValidationRule,
)


class ValidationRepository:
    """Repository for validation data access."""

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def create_validation(self, validation: Validation) -> Validation:
        """Create a new validation."""
        self.db.add(validation)
        await self.db.commit()
        await self.db.refresh(validation)
        return validation

    async def get_validation_by_id(self, validation_id: int) -> Validation | None:
        """Get validation by ID."""
        stmt = (
            select(Validation)
            .options(
                selectinload(Validation.analysis),
                selectinload(Validation.validator),
            )
            .where(Validation.id == validation_id)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_validation_by_analysis(self, analysis_id: int) -> Validation | None:
        """Get validation by analysis ID."""
        stmt = (
            select(Validation)
            .options(
                selectinload(Validation.analysis),
                selectinload(Validation.validator),
            )
            .where(Validation.analysis_id == analysis_id)
        )
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_validations_by_analysis(self, analysis_id: int) -> list[Validation]:
        """Get all validations for an analysis."""
        stmt = (
            select(Validation)
            .options(
                selectinload(Validation.validator),
            )
            .where(Validation.analysis_id == analysis_id)
            .order_by(desc(Validation.created_at))
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_validations_by_validator(
        self, validator_id: int, limit: int = 50, offset: int = 0
    ) -> list[Validation]:
        """Get validations by validator."""
        stmt = (
            select(Validation)
            .options(
                selectinload(Validation.analysis),
            )
            .where(Validation.validator_id == validator_id)
            .order_by(desc(Validation.created_at))
            .limit(limit)
            .offset(offset)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def update_validation(
        self, validation_id: int, update_data: dict[str, Any]
    ) -> Validation | None:
        """Update validation."""
        stmt = select(Validation).where(Validation.id == validation_id)
        result = await self.db.execute(stmt)
        validation = result.scalar_one_or_none()

        if validation:
            for key, value in update_data.items():
                if hasattr(validation, key):
                    setattr(validation, key, value)

            await self.db.commit()
            await self.db.refresh(validation)

        return validation

    async def get_available_validators(
        self,
        min_role: UserRoles = UserRoles.TECHNICIAN,
        min_experience_years: int | None = None,
    ) -> list[User]:
        """Get available validators based on criteria."""
        stmt = (
            select(User)
            .where(User.is_active.is_(True))
            .where(
                User.role.in_(
                    [
                        UserRoles.TECHNICIAN,
                        UserRoles.PHYSICIAN,
                        UserRoles.CARDIOLOGIST,
                        UserRoles.ADMIN,
                    ]
                )
            )
        )

        if min_role == UserRoles.CARDIOLOGIST:
            stmt = stmt.where(User.role.in_([UserRoles.CARDIOLOGIST, UserRoles.ADMIN]))
        elif min_role == UserRoles.PHYSICIAN:
            stmt = stmt.where(
                User.role.in_(
                    [UserRoles.PHYSICIAN, UserRoles.CARDIOLOGIST, UserRoles.ADMIN]
                )
            )

        if min_experience_years:
            stmt = stmt.where(
                and_(
                    User.experience_years.isnot(None),
                    User.experience_years >= min_experience_years,
                )
            )

        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def get_analysis_by_id(self, analysis_id: int) -> ECGAnalysis | None:
        """Get analysis by ID."""
        stmt = select(ECGAnalysis).where(ECGAnalysis.id == analysis_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def update_analysis_validation_status(
        self, analysis_id: int, is_validated: bool
    ) -> bool:
        """Update analysis validation status."""
        stmt = select(ECGAnalysis).where(ECGAnalysis.id == analysis_id)
        result = await self.db.execute(stmt)
        analysis = result.scalar_one_or_none()

        if analysis:
            analysis.is_validated = is_validated
            await self.db.commit()
            return True

        return False

    async def get_active_validation_rules(self) -> list[ValidationRule]:
        """Get active validation rules."""
        stmt = (
            select(ValidationRule)
            .where(ValidationRule.is_active.is_(True))
            .order_by(ValidationRule.name)
        )
        result = await self.db.execute(stmt)
        return list(result.scalars().all())

    async def create_validation_result(
        self, validation_result: ValidationResult
    ) -> ValidationResult:
        """Create validation result."""
        self.db.add(validation_result)
        await self.db.commit()
        await self.db.refresh(validation_result)
        return validation_result

    async def create_quality_metric(
        self, quality_metric: QualityMetric
    ) -> QualityMetric:
        """Create quality metric."""
        self.db.add(quality_metric)
        await self.db.commit()
        await self.db.refresh(quality_metric)
        return quality_metric

    async def get_validation_statistics(
        self, date_from: str | None = None, date_to: str | None = None
    ) -> dict[str, Any]:
        """Get validation statistics."""
        base_query = select(Validation)

        if date_from:
            base_query = base_query.where(Validation.created_at >= date_from)
        if date_to:
            base_query = base_query.where(Validation.created_at <= date_to)

        total_stmt = select(func.count(Validation.id))
        if date_from:
            total_stmt = total_stmt.where(Validation.created_at >= date_from)
        if date_to:
            total_stmt = total_stmt.where(Validation.created_at <= date_to)

        total_result = await self.db.execute(total_stmt)
        total_validations = total_result.scalar()

        status_stmt = select(Validation.status, func.count(Validation.id)).group_by(
            Validation.status
        )
        if date_from:
            status_stmt = status_stmt.where(Validation.created_at >= date_from)
        if date_to:
            status_stmt = status_stmt.where(Validation.created_at <= date_to)

        status_result = await self.db.execute(status_stmt)
        status_counts: dict[str, int] = {
            str(status): count for status, count in status_result.all()
        }

        avg_time_stmt = select(func.avg(Validation.validation_duration_minutes)).where(
            Validation.validation_duration_minutes.isnot(None)
        )
        if date_from:
            avg_time_stmt = avg_time_stmt.where(Validation.created_at >= date_from)
        if date_to:
            avg_time_stmt = avg_time_stmt.where(Validation.created_at <= date_to)

        avg_time_result = await self.db.execute(avg_time_stmt)
        avg_validation_time = avg_time_result.scalar()

        return {
            "total_validations": total_validations,
            "status_distribution": status_counts,
            "average_validation_time_minutes": (
                float(avg_validation_time) if avg_validation_time else None
            ),
            "approval_rate": (
                status_counts.get(str(ValidationStatus.APPROVED), 0)
                / max(total_validations or 1, 1)
                * 100
            ),
        }
