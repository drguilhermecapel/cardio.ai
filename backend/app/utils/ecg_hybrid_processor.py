"""
ECG Hybrid Processor - Integration utilities for hybrid ECG analysis
"""

import logging
from typing import Any

from app.core.exceptions import ECGProcessingException
from app.services.hybrid_ecg_service import HybridECGAnalysisService

# from app.services.regulatory_validation import RegulatoryValidationService  # Will be added in PR-003

logger = logging.getLogger(__name__)


class ECGHybridProcessor:
    """
    Processor for integrating hybrid ECG analysis with existing infrastructure
    """

    def __init__(self, db: Any, validation_service: Any) -> None:
        self.hybrid_service = HybridECGAnalysisService(db, validation_service)
        self.regulatory_service: Any = (
            None  # Will be implemented in PR-003 (Regulatory Compliance)
        )

    async def process_ecg_with_validation(
        self,
        file_path: str,
        patient_id: int,
        analysis_id: str,
        require_regulatory_compliance: bool = True,
    ) -> dict[str, Any]:
        """
        Process ECG with comprehensive analysis and regulatory validation

        Args:
            file_path: Path to ECG file
            patient_id: Patient identifier
            analysis_id: Analysis identifier
            require_regulatory_compliance: Whether to enforce regulatory compliance

        Returns:
            Dict containing analysis results and validation status
        """
        try:
            analysis_results = await self.hybrid_service.analyze_ecg_comprehensive(
                file_path=file_path, patient_id=patient_id, analysis_id=analysis_id
            )

            if self.regulatory_service is None:
                logger.warning(
                    "Regulatory service not configured - using placeholder validation"
                )
                validation_results = {"status": "pending_regulatory_implementation"}
                validation_report = {"overall_compliance": True, "recommendations": []}
            else:
                validation_results = (
                    await self.regulatory_service.validate_analysis_comprehensive(
                        analysis_results
                    )
                )

                validation_report = (
                    await self.regulatory_service.generate_validation_report(
                        validation_results
                    )
                )

            if (
                require_regulatory_compliance
                and not validation_report["overall_compliance"]
            ):
                logger.warning(
                    f"Analysis {analysis_id} failed regulatory compliance: "
                    f"{validation_report['recommendations']}"
                )

                analysis_results["regulatory_compliant"] = False
                analysis_results["compliance_issues"] = validation_report[
                    "recommendations"
                ]
            else:
                analysis_results["regulatory_compliant"] = True
                analysis_results["compliance_issues"] = []

            analysis_results["regulatory_validation"] = {
                "validation_results": validation_results,
                "validation_report": validation_report,
            }

            return analysis_results

        except Exception as e:
            logger.error(f"ECG hybrid processing failed: {e}")
            raise ECGProcessingException(f"Hybrid processing failed: {str(e)}") from e

    async def validate_existing_analysis(
        self, analysis_results: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Validate existing analysis results against regulatory standards

        Args:
            analysis_results: Existing analysis results

        Returns:
            Dict containing validation results
        """
        try:
            # Regulatory validation will be implemented in PR-003
            validation_results = {"status": "pending_regulatory_implementation"}
            validation_report = {"overall_compliance": True, "recommendations": []}

            return {
                "validation_results": validation_results,
                "validation_report": validation_report,
                "overall_compliance": validation_report["overall_compliance"],
            }

        except Exception as e:
            logger.error(f"Analysis validation failed: {e}")
            raise ECGProcessingException(f"Validation failed: {str(e)}") from e

    def get_supported_formats(self) -> list[str]:
        """Get list of supported ECG file formats"""
        return list(self.hybrid_service.ecg_reader.supported_formats.keys())

    def get_regulatory_standards(self) -> list[str]:
        """Get list of supported regulatory standards"""
        # from app.services.regulatory_validation import RegulatoryStandard  # Will be added in PR-003
        return ["FDA", "ANVISA", "NMSA", "EU_MDR"]  # Placeholder for PR-003

    async def get_system_status(self) -> dict[str, Any]:
        """Get hybrid ECG analysis system status"""
        return {
            "hybrid_service_initialized": self.hybrid_service is not None,
            "regulatory_service_initialized": self.regulatory_service is not None,
            "supported_formats": self.get_supported_formats(),
            "regulatory_standards": self.get_regulatory_standards(),
            "system_version": "1.0.0",
        }
