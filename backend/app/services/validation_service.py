"""
Validation Service - Medical validation and quality control.
"""

import logging
from datetime import datetime
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.constants import ClinicalUrgency, UserRoles, ValidationStatus
from app.core.exceptions import PermissionDeniedException, ValidationException
from app.models.ecg_analysis import ECGAnalysis
from app.models.validation import (
    QualityMetric,
    Validation,
    ValidationResult,
    ValidationRule,
)
from app.repositories.validation_repository import ValidationRepository
from app.services.notification_service import NotificationService

logger = logging.getLogger(__name__)


class ValidationService:
    """Service for medical validation and quality control."""

    def __init__(
        self,
        db: AsyncSession,
        notification_service: NotificationService,
    ) -> None:
        self.db = db
        self.repository = ValidationRepository(db)
        self.notification_service = notification_service

    async def create_validation(
        self,
        analysis_id: int,
        validator_id: int,
        validator_role: UserRoles,
        validator_experience_years: int | None = None,
    ) -> Validation:
        """Create a new validation."""
        try:
            existing = await self.repository.get_validation_by_analysis(analysis_id)
            if existing:
                raise ValidationException("Validation already exists for this analysis")

            analysis = await self.repository.get_analysis_by_id(analysis_id)
            if not analysis:
                raise ValidationException("Analysis not found")

            if not self._can_validate(
                validator_role, validator_experience_years, analysis.clinical_urgency
            ):
                raise PermissionDeniedException(
                    "Insufficient permissions to validate this analysis"
                )

            validation = Validation()
            validation.analysis_id = analysis_id
            validation.validator_id = validator_id
            validation.status = ValidationStatus.PENDING
            validation.requires_second_opinion = self._requires_second_opinion(analysis)

            validation = await self.repository.create_validation(validation)

            await self.notification_service.send_validation_assignment(
                validator_id, analysis_id, analysis.clinical_urgency
            )

            logger.info(
                f"Validation created: validation_id={validation.id}, analysis_id={analysis_id}, validator_id={validator_id}"
            )

            return validation

        except Exception as e:
            logger.error(
                f"Failed to create validation: error={str(e)}, analysis_id={analysis_id}, validator_id={validator_id}"
            )
            raise

    async def submit_validation(
        self,
        validation_id: int,
        validator_id: int,
        validation_data: dict[str, Any],
    ) -> Validation:
        """Submit validation results."""
        try:
            validation = await self.repository.get_validation_by_id(validation_id)
            if not validation:
                raise ValidationException("Validation not found")

            if validation.validator_id != validator_id:
                raise PermissionDeniedException(
                    "Not authorized to submit this validation"
                )

            if validation.status != ValidationStatus.PENDING:
                raise ValidationException("Validation already completed")

            update_data = {
                "status": (
                    ValidationStatus.APPROVED
                    if validation_data.get("approved", True)
                    else ValidationStatus.REJECTED
                ),
                "validation_date": datetime.utcnow(),
                "agrees_with_ai": validation_data.get("agrees_with_ai"),
                "clinical_notes": validation_data.get("clinical_notes"),
                "corrected_diagnosis": validation_data.get("corrected_diagnosis"),
                "corrected_urgency": validation_data.get("corrected_urgency"),
                "signal_quality_rating": validation_data.get("signal_quality_rating"),
                "ai_confidence_rating": validation_data.get("ai_confidence_rating"),
                "overall_quality_score": validation_data.get("overall_quality_score"),
                "recommendations": validation_data.get("recommendations", []),
                "follow_up_required": validation_data.get("follow_up_required", False),
                "follow_up_notes": validation_data.get("follow_up_notes"),
                "validation_duration_minutes": validation_data.get(
                    "validation_duration_minutes"
                ),
            }

            if validation_data.get("digital_signature"):
                update_data["digital_signature"] = validation_data["digital_signature"]
                update_data["signature_timestamp"] = datetime.utcnow()

            updated_validation = await self.repository.update_validation(
                validation_id, update_data
            )
            if not updated_validation:
                raise ValidationException("Failed to update validation")

            await self._update_analysis_validation_status(
                updated_validation.analysis_id
            )

            await self._calculate_quality_metrics(
                updated_validation.analysis_id, validation_data
            )

            await self._send_validation_notifications(updated_validation)

            validation = updated_validation

            logger.info(
                f"Validation submitted: validation_id={validation_id}, status={validation.status}, validator_id={validator_id}"
            )

            return validation

        except Exception as e:
            logger.error(
                f"Failed to submit validation: error={str(e)}, validation_id={validation_id}, validator_id={validator_id}"
            )
            raise

    async def create_urgent_validation(self, analysis_id: int) -> None:
        """Create urgent validation for critical findings."""
        try:
            available_validators = await self.repository.get_available_validators(
                min_role=UserRoles.CARDIOLOGIST,
                min_experience_years=settings.MIN_EXPERIENCE_YEARS_CRITICAL,
            )

            if not available_validators:
                available_validators = await self.repository.get_available_validators(
                    min_role=UserRoles.PHYSICIAN,
                    min_experience_years=3,
                )

            if available_validators:
                validator = max(
                    available_validators, key=lambda v: v.experience_years or 0
                )

                await self.create_validation(
                    analysis_id=analysis_id,
                    validator_id=validator.id,
                    validator_role=validator.role,
                    validator_experience_years=validator.experience_years,
                )

                await self.notification_service.send_urgent_validation_alert(
                    validator.id, analysis_id
                )
            else:
                await self.notification_service.send_no_validator_alert(analysis_id)

                logger.warning(
                    f"No validators available for urgent validation: analysis_id={analysis_id}"
                )

        except Exception as e:
            logger.error(
                f"Failed to create urgent validation: error={str(e)}, analysis_id={analysis_id}"
            )

    async def run_automated_validation_rules(
        self, analysis_id: int
    ) -> list[ValidationResult]:
        """Run automated validation rules on analysis."""
        try:
            rules = await self.repository.get_active_validation_rules()

            analysis = await self.repository.get_analysis_by_id(analysis_id)
            if not analysis:
                raise ValidationException("Analysis not found")

            results = []

            for rule in rules:
                try:
                    result = await self._execute_validation_rule(rule, analysis)
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(
                        f"Validation rule execution failed: rule_id={rule.id}, rule_name={rule.name}, error={str(e)}"
                    )

            logger.info(
                f"Automated validation completed: analysis_id={analysis_id}, rules_executed={len(rules)}, results_count={len(results)}"
            )

            return results

        except Exception as e:
            logger.error(
                f"Automated validation failed: error={str(e)}, analysis_id={analysis_id}"
            )
            return []

    def _can_validate(
        self,
        validator_role: UserRoles,
        validator_experience_years: int | None,
        clinical_urgency: ClinicalUrgency,
    ) -> bool:
        """Check if validator can validate based on role and experience."""
        if validator_role == UserRoles.ADMIN:
            return True

        if clinical_urgency == ClinicalUrgency.CRITICAL:
            if validator_role == UserRoles.CARDIOLOGIST:
                return True
            if (
                validator_role == UserRoles.PHYSICIAN
                and validator_experience_years
                and validator_experience_years >= settings.MIN_EXPERIENCE_YEARS_CRITICAL
            ):
                return True
            return False

        if clinical_urgency == ClinicalUrgency.HIGH:
            return validator_role in [UserRoles.PHYSICIAN, UserRoles.CARDIOLOGIST]

        return validator_role in [
            UserRoles.TECHNICIAN,
            UserRoles.PHYSICIAN,
            UserRoles.CARDIOLOGIST,
        ]

    def _requires_second_opinion(self, analysis: ECGAnalysis) -> bool:
        """Determine if analysis requires second opinion."""
        if not settings.REQUIRE_DOUBLE_VALIDATION_CRITICAL:
            return False

        if analysis.clinical_urgency == ClinicalUrgency.CRITICAL:
            return True
        if analysis.requires_immediate_attention:
            return True
        ai_confidence = getattr(analysis, "ai_confidence", None)
        if ai_confidence is not None and ai_confidence < 0.7:
            return True
        return False

    async def _update_analysis_validation_status(self, analysis_id: int) -> None:
        """Update analysis validation status."""
        try:
            validations = await self.repository.get_validations_by_analysis(analysis_id)

            completed_validations = [
                v
                for v in validations
                if v.status in [ValidationStatus.APPROVED, ValidationStatus.REJECTED]
            ]

            is_validated = len(completed_validations) > 0

            await self.repository.update_analysis_validation_status(
                analysis_id, is_validated
            )

        except Exception as e:
            logger.error(
                f"Failed to update analysis validation status: error={str(e)}, analysis_id={analysis_id}"
            )

    async def _calculate_quality_metrics(
        self, analysis_id: int, validation_data: dict[str, Any]
    ) -> None:
        """Calculate and store quality metrics."""
        try:
            metrics = []

            if validation_data.get("signal_quality_rating"):
                metric = QualityMetric()
                metric.analysis_id = analysis_id
                metric.metric_name = "signal_quality_rating"
                metric.metric_value = float(validation_data["signal_quality_rating"])
                metric.metric_unit = "score"
                metric.normal_min = 3.0
                metric.normal_max = 5.0
                metric.is_within_normal = validation_data["signal_quality_rating"] >= 3
                metric.calculation_method = "manual_validation"
                metrics.append(metric)

            if validation_data.get("ai_confidence_rating"):
                metric = QualityMetric()
                metric.analysis_id = analysis_id
                metric.metric_name = "ai_confidence_rating"
                metric.metric_value = float(validation_data["ai_confidence_rating"])
                metric.metric_unit = "score"
                metric.normal_min = 3.0
                metric.normal_max = 5.0
                metric.is_within_normal = validation_data["ai_confidence_rating"] >= 3
                metric.calculation_method = "manual_validation"
                metrics.append(metric)

            if validation_data.get("overall_quality_score"):
                metric = QualityMetric()
                metric.analysis_id = analysis_id
                metric.metric_name = "overall_quality_score"
                metric.metric_value = float(validation_data["overall_quality_score"])
                metric.metric_unit = "score"
                metric.normal_min = 0.7
                metric.normal_max = 1.0
                metric.is_within_normal = (
                    validation_data["overall_quality_score"] >= 0.7
                )
                metric.calculation_method = "manual_validation"
                metrics.append(metric)

            for metric in metrics:
                await self.repository.create_quality_metric(metric)

        except Exception as e:
            logger.error(
                f"Failed to calculate quality metrics: error={str(e)}, analysis_id={analysis_id}"
            )

    async def _execute_validation_rule(
        self, rule: ValidationRule, analysis: ECGAnalysis
    ) -> ValidationResult | None:
        """Execute a single validation rule."""
        try:
            start_time = datetime.utcnow()

            if rule.rule_type == "threshold":
                result = await self._execute_threshold_rule(rule, analysis)
            elif rule.rule_type == "pattern":
                result = await self._execute_pattern_rule(rule, analysis)
            elif rule.rule_type == "ml":
                result = await self._execute_ml_rule(rule, analysis)
            else:
                logger.warning(f"Unknown rule type: {rule.rule_type}")
                return None

            end_time = datetime.utcnow()
            execution_time_ms = int((end_time - start_time).total_seconds() * 1000)

            validation_result = ValidationResult()
            validation_result.analysis_id = analysis.id
            validation_result.rule_id = rule.id
            validation_result.passed = result["passed"]
            validation_result.score = result.get("score")
            validation_result.message = result["message"]
            validation_result.details = result.get("details", {})
            validation_result.execution_time_ms = execution_time_ms

            return await self.repository.create_validation_result(validation_result)

        except Exception as e:
            logger.error(f"Rule execution failed: rule_id={rule.id}, error={str(e)}")
            return None

    async def _execute_threshold_rule(
        self, rule: ValidationRule, analysis: ECGAnalysis
    ) -> dict[str, Any]:
        """Execute threshold-based validation rule."""
        parameters = rule.parameters
        field = parameters.get("field")
        min_value = parameters.get("min_value")
        max_value = parameters.get("max_value")

        if not field or not hasattr(analysis, field):
            return {
                "passed": False,
                "message": f"Invalid field: {field}",
            }

        value = getattr(analysis, field)
        if value is None:
            return {
                "passed": False,
                "message": f"Missing value for {field}",
            }

        passed = True
        issues = []

        if min_value is not None and value < min_value:
            passed = False
            issues.append(f"{field} ({value}) below minimum ({min_value})")

        if max_value is not None and value > max_value:
            passed = False
            issues.append(f"{field} ({value}) above maximum ({max_value})")

        return {
            "passed": passed,
            "message": "; ".join(issues) if issues else f"{field} within normal range",
            "score": 1.0 if passed else 0.0,
            "details": {
                "field": field,
                "value": value,
                "min_value": min_value,
                "max_value": max_value,
            },
        }

    async def _execute_pattern_rule(
        self, rule: ValidationRule, analysis: ECGAnalysis
    ) -> dict[str, Any]:
        """Execute pattern-based validation rule."""
        return {
            "passed": True,
            "message": "Pattern validation not implemented",
            "score": 0.5,
        }

    async def _execute_ml_rule(
        self, rule: ValidationRule, analysis: ECGAnalysis
    ) -> dict[str, Any]:
        """Execute ML-based validation rule."""
        return {
            "passed": True,
            "message": "ML validation not implemented",
            "score": 0.5,
        }

    async def _send_validation_notifications(self, validation: Validation) -> None:
        """Send notifications after validation completion."""
        try:
            if validation.analysis and validation.analysis.created_by:
                await self.notification_service.send_validation_complete(
                    validation.analysis.created_by,
                    validation.analysis_id,
                    validation.status,
                )

            if (
                validation.status == ValidationStatus.REJECTED
                and validation.analysis
                and validation.analysis.clinical_urgency == ClinicalUrgency.CRITICAL
            ):
                await self.notification_service.send_critical_rejection_alert(
                    validation.analysis_id
                )

        except Exception as e:
            logger.error(
                f"Failed to send validation notifications: error={str(e)}, validation_id={validation.id}"
            )

    async def get_validation_by_id(self, validation_id: int) -> Any:
        """Get validation by ID."""
        return await self.repository.get_validation_by_id(validation_id)
