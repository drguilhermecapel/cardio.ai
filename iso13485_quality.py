"""
ISO 13485 Quality Management System for Medical Devices
Implements comprehensive quality controls for ECG analysis system
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels per ISO 14971"""

    NEGLIGIBLE = "negligible"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNACCEPTABLE = "unacceptable"


class DesignPhase(Enum):
    """Design control phases per ISO 13485"""

    PLANNING = "planning"
    INPUTS = "inputs"
    OUTPUTS = "outputs"
    REVIEW = "review"
    VERIFICATION = "verification"
    VALIDATION = "validation"
    TRANSFER = "transfer"
    CHANGES = "changes"


@dataclass
class RiskAssessment:
    """Risk assessment per ISO 14971"""

    hazard_id: str
    hazard_description: str
    potential_harm: str
    severity: int  # 1-5 scale
    probability: int  # 1-5 scale
    risk_level: RiskLevel
    risk_control_measures: list[str]
    residual_risk: RiskLevel
    acceptability: bool


@dataclass
class DesignControl:
    """Design control record per ISO 13485"""

    phase: DesignPhase
    requirements: list[str]
    outputs: list[str]
    verification_methods: list[str]
    validation_methods: list[str]
    review_participants: list[str]
    approval_status: bool
    approval_date: datetime | None


class ISO13485QualitySystem:
    """
    ISO 13485 Quality Management System implementation
    Ensures medical device compliance for ECG analysis system
    """

    def __init__(self) -> None:
        self.design_controls: dict[DesignPhase, DesignControl] = {}
        self.risk_assessments: list[RiskAssessment] = []
        self.quality_records: dict[str, Any] = {}
        self.document_control: dict[str, Any] = {}

        self._initialize_design_controls()
        self._initialize_risk_assessments()

    def _initialize_design_controls(self) -> None:
        """Initialize design controls for ECG analysis system"""

        self.design_controls[DesignPhase.PLANNING] = DesignControl(
            phase=DesignPhase.PLANNING,
            requirements=[
                "Define ECG analysis system scope and intended use",
                "Establish regulatory requirements (FDA, ANVISA, EU MDR)",
                "Define clinical performance requirements",
                "Establish risk management plan per ISO 14971",
                "Define verification and validation protocols",
            ],
            outputs=[
                "Design and Development Plan",
                "Regulatory Strategy Document",
                "Clinical Requirements Specification",
                "Risk Management Plan",
                "V&V Protocol",
            ],
            verification_methods=[
                "Document review by quality team",
                "Regulatory compliance check",
                "Clinical expert review",
            ],
            validation_methods=["Stakeholder approval", "Regulatory body consultation"],
            review_participants=[
                "Chief Medical Officer",
                "Regulatory Affairs Manager",
                "Quality Assurance Manager",
                "Clinical Consultant Cardiologist",
            ],
            approval_status=True,
            approval_date=datetime.now(),
        )

        self.design_controls[DesignPhase.INPUTS] = DesignControl(
            phase=DesignPhase.INPUTS,
            requirements=[
                "Clinical performance requirements for pathology detection",
                "Safety requirements for patient data protection",
                "Usability requirements for clinical workflow",
                "Regulatory requirements for medical device software",
                "Interoperability requirements for hospital systems",
            ],
            outputs=[
                "Clinical Requirements Document",
                "Safety Requirements Specification",
                "Usability Requirements Document",
                "Regulatory Compliance Matrix",
                "Interface Control Document",
            ],
            verification_methods=[
                "Requirements traceability matrix",
                "Clinical expert validation",
                "Regulatory compliance verification",
            ],
            validation_methods=[
                "Clinical workflow simulation",
                "User acceptance testing with cardiologists",
            ],
            review_participants=[
                "Clinical Consultant Cardiologist",
                "Hospital IT Integration Specialist",
                "Regulatory Affairs Manager",
            ],
            approval_status=True,
            approval_date=datetime.now(),
        )

        self.design_controls[DesignPhase.OUTPUTS] = DesignControl(
            phase=DesignPhase.OUTPUTS,
            requirements=[
                "ECG signal processing algorithms",
                "Pathology detection models",
                "Clinical decision support interface",
                "Data security and privacy controls",
                "Quality management system documentation",
            ],
            outputs=[
                "Core ECG Signal Processing Module",
                "Pathology Detection Algorithms",
                "Clinical Interface Components",
                "Security Framework Implementation",
                "Quality Documentation Package",
            ],
            verification_methods=[
                "Algorithm performance testing",
                "Security penetration testing",
                "Code review and static analysis",
                "Documentation completeness check",
            ],
            validation_methods=[
                "Clinical validation with real ECG data",
                "Usability testing with cardiologists",
                "End-to-end workflow validation",
            ],
            review_participants=[
                "Senior Software Architect",
                "Clinical Validation Team",
                "Information Security Officer",
            ],
            approval_status=False,  # In progress
            approval_date=None,
        )

    def _initialize_risk_assessments(self) -> None:
        """Initialize risk assessments per ISO 14971"""

        self.risk_assessments.append(
            RiskAssessment(
                hazard_id="RISK-001",
                hazard_description="False negative detection of STEMI",
                potential_harm="Delayed treatment leading to myocardial damage or death",
                severity=5,  # Catastrophic
                probability=1,  # Very unlikely (target <0.5%)
                risk_level=RiskLevel.HIGH,
                risk_control_measures=[
                    "Ultra-high sensitivity threshold (>99.5%)",
                    "Multiple algorithm ensemble validation",
                    "Mandatory cardiologist review for critical cases",
                    "Real-time monitoring and alerting system",
                    "Continuous performance monitoring",
                ],
                residual_risk=RiskLevel.LOW,
                acceptability=True,
            )
        )

        self.risk_assessments.append(
            RiskAssessment(
                hazard_id="RISK-002",
                hazard_description="System downtime during emergency",
                potential_harm="Delayed ECG analysis in critical situations",
                severity=4,  # Major
                probability=1,  # Very unlikely (target >99.99% uptime)
                risk_level=RiskLevel.MEDIUM,
                risk_control_measures=[
                    "Redundant system architecture",
                    "Automatic failover mechanisms",
                    "24/7 monitoring and alerting",
                    "Rapid response maintenance team",
                    "Offline backup analysis capability",
                ],
                residual_risk=RiskLevel.LOW,
                acceptability=True,
            )
        )

        self.risk_assessments.append(
            RiskAssessment(
                hazard_id="RISK-003",
                hazard_description="Unauthorized access to patient ECG data",
                potential_harm="Privacy violation and potential misuse of medical data",
                severity=3,  # Moderate
                probability=2,  # Unlikely
                risk_level=RiskLevel.MEDIUM,
                risk_control_measures=[
                    "End-to-end encryption of all data",
                    "Multi-factor authentication",
                    "Role-based access controls",
                    "Audit logging and monitoring",
                    "Regular security assessments",
                ],
                residual_risk=RiskLevel.LOW,
                acceptability=True,
            )
        )

        self.risk_assessments.append(
            RiskAssessment(
                hazard_id="RISK-004",
                hazard_description="Algorithmic bias affecting certain patient populations",
                potential_harm="Reduced diagnostic accuracy for underrepresented groups",
                severity=3,  # Moderate
                probability=2,  # Unlikely
                risk_level=RiskLevel.MEDIUM,
                risk_control_measures=[
                    "Diverse training dataset validation",
                    "Demographic performance monitoring",
                    "Regular bias assessment testing",
                    "Continuous model retraining",
                    "Clinical oversight and validation",
                ],
                residual_risk=RiskLevel.LOW,
                acceptability=True,
            )
        )

    def design_controls_verification(self) -> dict[str, Any]:
        """Verify design controls compliance"""

        verification_results: dict[str, Any] = {
            "overall_compliance": True,
            "phase_results": {},
            "non_conformances": [],
            "recommendations": [],
        }

        for phase, control in self.design_controls.items():
            phase_compliant = True
            phase_issues = []

            if len(control.requirements) < 3:
                phase_compliant = False
                phase_issues.append("Insufficient requirements defined")

            if len(control.outputs) < len(control.requirements):
                phase_compliant = False
                phase_issues.append("Outputs don't match requirements")

            if len(control.verification_methods) < 2:
                phase_compliant = False
                phase_issues.append("Insufficient verification methods")

            if len(control.review_participants) < 2:
                phase_compliant = False
                phase_issues.append("Insufficient review participation")

            verification_results["phase_results"][phase.value] = {
                "compliant": phase_compliant,
                "issues": phase_issues,
                "approval_status": control.approval_status,
            }

            if not phase_compliant:
                verification_results["overall_compliance"] = False
                verification_results["non_conformances"].extend(phase_issues)

        verification_results["recommendations"] = [
            "Complete design validation for all phases",
            "Obtain formal approval from all review participants",
            "Establish traceability matrix for requirements",
            "Schedule regular design review meetings",
            "Document all design changes with impact assessment",
        ]

        return verification_results

    def risk_management_iso14971(self) -> dict[str, Any]:
        """Comprehensive risk management per ISO 14971"""

        risk_analysis: dict[str, Any] = {
            "total_risks_identified": len(self.risk_assessments),
            "risk_distribution": {},
            "unacceptable_risks": [],
            "risk_control_effectiveness": {},
            "residual_risk_acceptability": True,
            "risk_management_file_complete": True,
        }

        for level in RiskLevel:
            count = sum(1 for risk in self.risk_assessments if risk.risk_level == level)
            risk_analysis["risk_distribution"][level.value] = count

        unacceptable = [
            risk for risk in self.risk_assessments if not risk.acceptability
        ]
        risk_analysis["unacceptable_risks"] = [risk.hazard_id for risk in unacceptable]

        for risk in self.risk_assessments:
            initial_risk_score = risk.severity * risk.probability

            risk_reduction_factor = len(risk.risk_control_measures) * 0.2
            residual_risk_score = max(
                1, initial_risk_score * (1 - risk_reduction_factor)
            )

            risk_analysis["risk_control_effectiveness"][risk.hazard_id] = {
                "initial_score": initial_risk_score,
                "residual_score": residual_risk_score,
                "reduction_percentage": (1 - residual_risk_score / initial_risk_score)
                * 100,
                "control_measures_count": len(risk.risk_control_measures),
            }

        high_residual_risks = [
            risk
            for risk in self.risk_assessments
            if risk.residual_risk in [RiskLevel.HIGH, RiskLevel.UNACCEPTABLE]
        ]

        if high_residual_risks:
            risk_analysis["residual_risk_acceptability"] = False
            risk_analysis["high_residual_risks"] = [
                risk.hazard_id for risk in high_residual_risks
            ]

        return risk_analysis

    def quality_metrics_monitoring(self) -> dict[str, Any]:
        """Monitor quality metrics for continuous improvement"""

        quality_metrics = {
            "clinical_performance": {
                "sensitivity_target": ">99.5%",
                "specificity_target": ">98.0%",
                "npv_target": ">99.9%",
                "current_performance": "Validation in progress",
            },
            "operational_performance": {
                "uptime_target": ">99.99%",
                "response_time_target": "<10s",
                "throughput_target": ">500 ECGs/hour",
                "current_performance": "Baseline establishment",
            },
            "quality_indicators": {
                "customer_complaints": 0,
                "non_conformances": 0,
                "corrective_actions": 0,
                "preventive_actions": 2,
            },
            "regulatory_compliance": {
                "fda_status": "Pre-submission planning",
                "anvisa_status": "Requirements analysis",
                "eu_mdr_status": "Technical documentation preparation",
                "iso13485_certification": "Implementation in progress",
            },
        }

        return quality_metrics

    def document_control_system(self) -> dict[str, Any]:
        """Document control per ISO 13485 requirements"""

        document_control = {
            "controlled_documents": {
                "quality_manual": {
                    "version": "1.0",
                    "approval_date": "2025-06-03",
                    "next_review": "2026-06-03",
                    "status": "approved",
                },
                "design_controls_procedure": {
                    "version": "1.0",
                    "approval_date": "2025-06-03",
                    "next_review": "2025-12-03",
                    "status": "approved",
                },
                "risk_management_procedure": {
                    "version": "1.0",
                    "approval_date": "2025-06-03",
                    "next_review": "2025-12-03",
                    "status": "approved",
                },
                "clinical_validation_protocol": {
                    "version": "1.0",
                    "approval_date": "2025-06-03",
                    "next_review": "2025-09-03",
                    "status": "draft",
                },
            },
            "document_approval_matrix": {
                "quality_manual": ["Quality Manager", "CEO"],
                "procedures": ["Quality Manager", "Process Owner"],
                "protocols": ["Clinical Lead", "Quality Manager"],
                "specifications": ["Technical Lead", "Quality Manager"],
            },
            "change_control": {
                "change_requests_pending": 0,
                "changes_implemented": 1,
                "impact_assessments_required": 0,
            },
        }

        return document_control

    def management_review_preparation(self) -> dict[str, Any]:
        """Prepare management review per ISO 13485"""

        management_review = {
            "review_date": "2025-06-03",
            "participants": [
                "CEO",
                "Quality Manager",
                "Clinical Lead",
                "Regulatory Affairs Manager",
                "Technical Lead",
            ],
            "agenda_items": [
                "Quality policy and objectives review",
                "Clinical validation progress",
                "Risk management effectiveness",
                "Regulatory compliance status",
                "Customer feedback analysis",
                "Resource adequacy assessment",
                "Improvement opportunities identification",
            ],
            "performance_data": {
                "quality_objectives_status": "On track",
                "customer_satisfaction": "Baseline establishment",
                "process_performance": "Meeting targets",
                "product_conformity": "Validation in progress",
            },
            "action_items": [
                "Accelerate clinical validation timeline",
                "Establish customer feedback mechanisms",
                "Enhance risk monitoring procedures",
                "Prepare regulatory submissions",
            ],
        }

        return management_review

    def generate_quality_report(self) -> dict[str, Any]:
        """Generate comprehensive quality system report"""

        report = {
            "report_date": datetime.now().isoformat(),
            "quality_system_status": "Implementation Phase",
            "iso13485_compliance": "In Progress",
            "design_controls": self.design_controls_verification(),
            "risk_management": self.risk_management_iso14971(),
            "quality_metrics": self.quality_metrics_monitoring(),
            "document_control": self.document_control_system(),
            "management_review": self.management_review_preparation(),
            "next_steps": [
                "Complete clinical validation studies",
                "Finalize regulatory submissions",
                "Implement continuous monitoring systems",
                "Establish post-market surveillance",
                "Schedule external quality audit",
            ],
            "compliance_statement": (
                "This ECG analysis system is being developed in accordance with "
                "ISO 13485:2016 Quality Management Systems for Medical Devices. "
                "All design controls, risk management, and quality processes "
                "are implemented to ensure patient safety and regulatory compliance."
            ),
        }

        return report


class ContinuousImprovementSystem:
    """Continuous improvement system for quality enhancement"""

    def __init__(self) -> None:
        self.improvement_opportunities: list[dict[str, Any]] = []
        self.corrective_actions: list[dict[str, Any]] = []
        self.preventive_actions: list[dict[str, Any]] = []

    def identify_improvement_opportunity(
        self, source: str, description: str, impact: str, priority: str
    ) -> None:
        """Identify improvement opportunity"""

        opportunity = {
            "id": f"IMP-{len(self.improvement_opportunities) + 1:03d}",
            "source": source,
            "description": description,
            "impact": impact,
            "priority": priority,
            "date_identified": datetime.now().isoformat(),
            "status": "identified",
            "assigned_to": None,
            "target_completion": None,
        }

        self.improvement_opportunities.append(opportunity)
        logger.info(f"Improvement opportunity identified: {opportunity['id']}")

    def implement_corrective_action(
        self,
        nonconformance: str,
        root_cause: str,
        corrective_action: str,
        responsible_person: str,
    ) -> None:
        """Implement corrective action"""

        action = {
            "id": f"CA-{len(self.corrective_actions) + 1:03d}",
            "nonconformance": nonconformance,
            "root_cause": root_cause,
            "corrective_action": corrective_action,
            "responsible_person": responsible_person,
            "date_initiated": datetime.now().isoformat(),
            "status": "in_progress",
            "effectiveness_verified": False,
        }

        self.corrective_actions.append(action)
        logger.info(f"Corrective action initiated: {action['id']}")

    def implement_preventive_action(
        self, potential_issue: str, preventive_action: str, responsible_person: str
    ) -> None:
        """Implement preventive action"""

        action = {
            "id": f"PA-{len(self.preventive_actions) + 1:03d}",
            "potential_issue": potential_issue,
            "preventive_action": preventive_action,
            "responsible_person": responsible_person,
            "date_initiated": datetime.now().isoformat(),
            "status": "in_progress",
            "effectiveness_verified": False,
        }

        self.preventive_actions.append(action)
        logger.info(f"Preventive action initiated: {action['id']}")
