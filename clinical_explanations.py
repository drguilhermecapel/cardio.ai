"""
Clinical Explanation Generator for ECG diagnoses
Provides automated text explanations with diagnostic criteria references
Based on scientific recommendations for CardioAI Pro
"""

import logging
from dataclasses import dataclass
from typing import Any

from app.core.scp_ecg_conditions import get_condition_by_code

logger = logging.getLogger(__name__)


@dataclass
class NormalRange:
    """Normal population range for ECG parameters"""

    parameter: str
    min_value: float
    max_value: float
    unit: str
    age_group: str = "adult"
    gender: str = "all"
    reference: str = "AHA/ACC/HRS Guidelines"


@dataclass
class DiagnosticCriterion:
    """Diagnostic criterion for ECG conditions"""

    condition_code: str
    criterion_name: str
    description: str
    threshold_value: float | None = None
    threshold_operator: str | None = None  # >, <, >=, <=, ==
    leads_affected: list[str] | None = None
    duration_requirement: str | None = None
    clinical_context: str | None = None
    reference_source: str = "Clinical Guidelines"


class ClinicalExplanationGenerator:
    """
    Generates clinical explanations for ECG diagnoses
    References established diagnostic criteria and normal population values
    """

    def __init__(self) -> None:
        self.diagnostic_criteria = self._load_diagnostic_criteria()
        self.normal_ranges = self._load_normal_ranges()
        self.lead_territories = self._load_lead_territories()

    def _load_diagnostic_criteria(self) -> dict[str, list[DiagnosticCriterion]]:
        """Load established diagnostic criteria for ECG conditions"""

        criteria = {}

        criteria["STEMI"] = [
            DiagnosticCriterion(
                condition_code="STEMI",
                criterion_name="ST Elevation - Limb Leads",
                description="ST elevation ≥1mm in at least 2 contiguous limb leads",
                threshold_value=1.0,
                threshold_operator=">=",
                leads_affected=["II", "III", "aVF", "I", "aVL"],
                reference_source="ESC/AHA STEMI Guidelines 2017",
            ),
            DiagnosticCriterion(
                condition_code="STEMI",
                criterion_name="ST Elevation - Precordial Leads",
                description="ST elevation ≥2mm in at least 2 contiguous precordial leads",
                threshold_value=2.0,
                threshold_operator=">=",
                leads_affected=["V1", "V2", "V3", "V4", "V5", "V6"],
                reference_source="ESC/AHA STEMI Guidelines 2017",
            ),
            DiagnosticCriterion(
                condition_code="STEMI",
                criterion_name="Duration Requirement",
                description="ST elevation persistent for >20 minutes",
                duration_requirement=">20 minutes",
                clinical_context="Chest pain or equivalent symptoms",
                reference_source="ESC/AHA STEMI Guidelines 2017",
            ),
        ]

        criteria["AFIB"] = [
            DiagnosticCriterion(
                condition_code="AFIB",
                criterion_name="Irregular RR Intervals",
                description="Irregularly irregular RR intervals",
                reference_source="AHA/ACC/HRS AF Guidelines 2019",
            ),
            DiagnosticCriterion(
                condition_code="AFIB",
                criterion_name="Absent P Waves",
                description="Absence of distinct P waves or presence of fibrillatory waves",
                reference_source="AHA/ACC/HRS AF Guidelines 2019",
            ),
            DiagnosticCriterion(
                condition_code="AFIB",
                criterion_name="Duration Requirement",
                description="Episode duration ≥30 seconds for diagnosis",
                duration_requirement="≥30 seconds",
                reference_source="AHA/ACC/HRS AF Guidelines 2019",
            ),
        ]

        criteria["LBBB"] = [
            DiagnosticCriterion(
                condition_code="LBBB",
                criterion_name="QRS Duration",
                description="QRS duration ≥120ms",
                threshold_value=120.0,
                threshold_operator=">=",
                reference_source="AHA/ACCF/HRS Guidelines 2009",
            ),
            DiagnosticCriterion(
                condition_code="LBBB",
                criterion_name="QS or rS Pattern",
                description="QS or rS pattern in leads V1-V3",
                leads_affected=["V1", "V2", "V3"],
                reference_source="AHA/ACCF/HRS Guidelines 2009",
            ),
            DiagnosticCriterion(
                condition_code="LBBB",
                criterion_name="Broad R Wave",
                description="Broad notched or slurred R wave in leads I, aVL, V5, V6",
                leads_affected=["I", "aVL", "V5", "V6"],
                reference_source="AHA/ACCF/HRS Guidelines 2009",
            ),
        ]

        criteria["RBBB"] = [
            DiagnosticCriterion(
                condition_code="RBBB",
                criterion_name="QRS Duration",
                description="QRS duration ≥120ms",
                threshold_value=120.0,
                threshold_operator=">=",
                reference_source="AHA/ACCF/HRS Guidelines 2009",
            ),
            DiagnosticCriterion(
                condition_code="RBBB",
                criterion_name="RSR Pattern",
                description="RSR', rsR', or rSR' pattern in leads V1 or V2",
                leads_affected=["V1", "V2"],
                reference_source="AHA/ACCF/HRS Guidelines 2009",
            ),
            DiagnosticCriterion(
                condition_code="RBBB",
                criterion_name="Wide S Wave",
                description="Wide S wave in leads I and V6",
                leads_affected=["I", "V6"],
                reference_source="AHA/ACCF/HRS Guidelines 2009",
            ),
        ]

        criteria["LVH"] = [
            DiagnosticCriterion(
                condition_code="LVH",
                criterion_name="Sokolow-Lyon Criteria",
                description="S wave in V1 + R wave in V5 or V6 ≥35mm",
                threshold_value=35.0,
                threshold_operator=">=",
                leads_affected=["V1", "V5", "V6"],
                reference_source="Sokolow & Lyon 1949",
            ),
            DiagnosticCriterion(
                condition_code="LVH",
                criterion_name="Cornell Criteria",
                description="R wave in aVL + S wave in V3 >28mm (men) or >20mm (women)",
                leads_affected=["aVL", "V3"],
                reference_source="Casale et al. 1985",
            ),
        ]

        criteria["VTAC"] = [
            DiagnosticCriterion(
                condition_code="VTAC",
                criterion_name="Heart Rate",
                description="Ventricular rate >100 bpm",
                threshold_value=100.0,
                threshold_operator=">",
                reference_source="AHA/ACC/HRS Guidelines 2017",
            ),
            DiagnosticCriterion(
                condition_code="VTAC",
                criterion_name="Wide QRS",
                description="QRS duration ≥120ms",
                threshold_value=120.0,
                threshold_operator=">=",
                reference_source="AHA/ACC/HRS Guidelines 2017",
            ),
            DiagnosticCriterion(
                condition_code="VTAC",
                criterion_name="AV Dissociation",
                description="AV dissociation (when visible)",
                clinical_context="Independent atrial and ventricular activity",
                reference_source="AHA/ACC/HRS Guidelines 2017",
            ),
        ]

        criteria["AVB3"] = [
            DiagnosticCriterion(
                condition_code="AVB3",
                criterion_name="AV Dissociation",
                description="Complete AV dissociation with independent P waves and QRS complexes",
                reference_source="AHA/ACCF/HRS Guidelines 2008",
            ),
            DiagnosticCriterion(
                condition_code="AVB3",
                criterion_name="Escape Rhythm",
                description="Ventricular escape rhythm typically 20-40 bpm",
                clinical_context="May be junctional (40-60 bpm) or ventricular (20-40 bpm)",
                reference_source="AHA/ACCF/HRS Guidelines 2008",
            ),
        ]

        return criteria

    def _load_normal_ranges(self) -> dict[str, NormalRange]:
        """Load normal population ranges for ECG parameters"""

        ranges = {}

        ranges["heart_rate"] = NormalRange(
            parameter="heart_rate",
            min_value=60.0,
            max_value=100.0,
            unit="bpm",
            reference="AHA/ACC Guidelines",
        )

        ranges["pr_interval"] = NormalRange(
            parameter="pr_interval",
            min_value=120.0,
            max_value=200.0,
            unit="ms",
            reference="AHA/ACC Guidelines",
        )

        ranges["qrs_duration"] = NormalRange(
            parameter="qrs_duration",
            min_value=80.0,
            max_value=120.0,
            unit="ms",
            reference="AHA/ACC Guidelines",
        )

        ranges["qt_interval"] = NormalRange(
            parameter="qt_interval",
            min_value=350.0,
            max_value=450.0,
            unit="ms",
            reference="AHA/ACC Guidelines",
        )

        ranges["qtc"] = NormalRange(
            parameter="qtc",
            min_value=350.0,
            max_value=440.0,
            unit="ms",
            reference="AHA/ACC Guidelines",
        )

        ranges["qrs_axis"] = NormalRange(
            parameter="qrs_axis",
            min_value=-30.0,
            max_value=90.0,
            unit="degrees",
            reference="AHA/ACC Guidelines",
        )

        return ranges

    def _load_lead_territories(self) -> dict[str, dict[str, Any]]:
        """Load anatomical territories for ECG leads"""

        territories = {
            "inferior": {
                "leads": ["II", "III", "aVF"],
                "artery": "Right Coronary Artery (RCA)",
                "anatomy": "Inferior wall of left ventricle",
            },
            "anterior": {
                "leads": ["V1", "V2", "V3", "V4"],
                "artery": "Left Anterior Descending (LAD)",
                "anatomy": "Anterior wall of left ventricle",
            },
            "lateral": {
                "leads": ["I", "aVL", "V5", "V6"],
                "artery": "Left Circumflex (LCX)",
                "anatomy": "Lateral wall of left ventricle",
            },
            "posterior": {
                "leads": ["V7", "V8", "V9"],
                "artery": "Posterior Descending Artery (PDA)",
                "anatomy": "Posterior wall of left ventricle",
            },
            "septal": {
                "leads": ["V1", "V2"],
                "artery": "Septal branches of LAD",
                "anatomy": "Interventricular septum",
            },
        }

        return territories

    def generate_explanation(
        self,
        features: dict[str, Any],
        predictions: dict[str, float],
        shap_explanation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate comprehensive clinical explanation"""

        try:
            primary_diagnosis = max(predictions.items(), key=lambda x: x[1])[0]
            confidence = predictions[primary_diagnosis]

            explanation = {
                "primary_diagnosis": self._explain_primary_diagnosis(
                    primary_diagnosis, confidence
                ),
                "diagnostic_criteria": self._explain_diagnostic_criteria(
                    primary_diagnosis, features
                ),
                "parameter_analysis": self._analyze_parameters(features),
                "clinical_significance": self._explain_clinical_significance(
                    primary_diagnosis, features
                ),
                "differential_diagnosis": self._generate_differential_diagnosis(
                    predictions
                ),
                "recommendations": self._generate_clinical_recommendations(
                    primary_diagnosis, features
                ),
                "anatomical_correlation": self._explain_anatomical_correlation(
                    primary_diagnosis, features, shap_explanation
                ),
                "risk_stratification": self._assess_risk_stratification(
                    primary_diagnosis, features
                ),
                "follow_up": self._recommend_follow_up(primary_diagnosis, features),
            }

            return explanation

        except Exception as e:
            logger.error(f"Error generating clinical explanation: {e}")
            return {
                "error": str(e),
                "primary_diagnosis": "Unable to generate explanation",
                "diagnostic_criteria": {},
                "parameter_analysis": {},
                "clinical_significance": "Analysis unavailable",
                "recommendations": [],
            }

    def _explain_primary_diagnosis(
        self, diagnosis_code: str, confidence: float
    ) -> dict[str, Any]:
        """Explain the primary diagnosis with confidence assessment"""

        condition = get_condition_by_code(diagnosis_code)

        explanation = {
            "code": diagnosis_code,
            "name": condition.name if condition else "Unknown condition",
            "confidence": confidence,
            "confidence_interpretation": self._interpret_confidence(confidence),
            "description": (
                condition.description if condition else "No description available"
            ),
            "clinical_urgency": condition.clinical_urgency if condition else "unknown",
        }

        if confidence >= 0.9:
            explanation["interpretation"] = (
                "High confidence diagnosis - findings strongly support this condition"
            )
        elif confidence >= 0.7:
            explanation["interpretation"] = (
                "Moderate confidence diagnosis - findings are consistent with this condition"
            )
        elif confidence >= 0.5:
            explanation["interpretation"] = (
                "Low confidence diagnosis - findings suggest this condition but require clinical correlation"
            )
        else:
            explanation["interpretation"] = (
                "Very low confidence - findings are inconclusive"
            )

        return explanation

    def _interpret_confidence(self, confidence: float) -> str:
        """Interpret confidence score in clinical terms"""

        if confidence >= 0.95:
            return "Very High (>95%)"
        elif confidence >= 0.85:
            return "High (85-95%)"
        elif confidence >= 0.70:
            return "Moderate (70-85%)"
        elif confidence >= 0.50:
            return "Low (50-70%)"
        else:
            return "Very Low (<50%)"

    def _explain_diagnostic_criteria(
        self, diagnosis_code: str, features: dict[str, Any]
    ) -> dict[str, Any]:
        """Explain how the diagnosis meets established criteria"""

        criteria_explanation: dict[str, Any] = {
            "condition": diagnosis_code,
            "criteria_met": [],
            "criteria_not_met": [],
            "partial_criteria": [],
            "reference_standards": [],
        }

        if diagnosis_code not in self.diagnostic_criteria:
            criteria_explanation["note"] = (
                f"No specific criteria loaded for {diagnosis_code}"
            )
            return criteria_explanation

        criteria_list = self.diagnostic_criteria[diagnosis_code]

        for criterion in criteria_list:
            criterion_assessment = self._assess_criterion(criterion, features)

            if criterion_assessment["met"]:
                criteria_explanation["criteria_met"].append(
                    {
                        "name": criterion.criterion_name,
                        "description": criterion.description,
                        "assessment": criterion_assessment["explanation"],
                        "reference": criterion.reference_source,
                    }
                )
            elif criterion_assessment["partial"]:
                criteria_explanation["partial_criteria"].append(
                    {
                        "name": criterion.criterion_name,
                        "description": criterion.description,
                        "assessment": criterion_assessment["explanation"],
                        "reference": criterion.reference_source,
                    }
                )
            else:
                criteria_explanation["criteria_not_met"].append(
                    {
                        "name": criterion.criterion_name,
                        "description": criterion.description,
                        "assessment": criterion_assessment["explanation"],
                        "reference": criterion.reference_source,
                    }
                )

        references = list({c.reference_source for c in criteria_list})
        criteria_explanation["reference_standards"] = references

        return criteria_explanation

    def _assess_criterion(
        self, criterion: DiagnosticCriterion, features: dict[str, Any]
    ) -> dict[str, Any]:
        """Assess whether a specific diagnostic criterion is met"""

        assessment = {"met": False, "partial": False, "explanation": "", "value": None}

        try:
            if criterion.threshold_value is not None and criterion.threshold_operator:
                feature_name = self._map_criterion_to_feature(criterion.criterion_name)

                if feature_name in features:
                    value = features[feature_name]
                    assessment["value"] = value

                    if criterion.threshold_operator == ">=":
                        met = value >= criterion.threshold_value
                    elif criterion.threshold_operator == ">":
                        met = value > criterion.threshold_value
                    elif criterion.threshold_operator == "<=":
                        met = value <= criterion.threshold_value
                    elif criterion.threshold_operator == "<":
                        met = value < criterion.threshold_value
                    elif criterion.threshold_operator == "==":
                        met = abs(value - criterion.threshold_value) < 0.01
                    else:
                        met = False

                    assessment["met"] = met
                    assessment["explanation"] = (
                        f"Measured value: {value:.1f}, Threshold: {criterion.threshold_operator}{criterion.threshold_value}"
                    )
                else:
                    assessment["explanation"] = (
                        f"Feature {feature_name} not available for assessment"
                    )

            else:
                assessment = self._assess_qualitative_criterion(criterion, features)

        except Exception as e:
            assessment["explanation"] = f"Error assessing criterion: {str(e)}"

        return assessment

    def _map_criterion_to_feature(self, criterion_name: str) -> str:
        """Map diagnostic criterion names to feature names"""

        mapping = {
            "QRS Duration": "qrs_duration",
            "Heart Rate": "heart_rate",
            "PR Interval": "pr_interval",
            "QT Interval": "qt_interval",
            "ST Elevation - Limb Leads": "st_elevation_max",
            "ST Elevation - Precordial Leads": "st_elevation_max",
            "QRS Axis": "qrs_axis",
        }

        return mapping.get(criterion_name, criterion_name.lower().replace(" ", "_"))

    def _assess_qualitative_criterion(
        self, criterion: DiagnosticCriterion, features: dict[str, Any]
    ) -> dict[str, Any]:
        """Assess qualitative diagnostic criteria"""

        assessment = {
            "met": False,
            "partial": False,
            "explanation": "Qualitative assessment not available with current features",
        }

        if "irregular" in criterion.description.lower():
            rr_std = features.get("rr_std", 0)
            if rr_std > 100:  # High RR variability suggests irregularity
                assessment["met"] = True
                assessment["explanation"] = (
                    f"High RR variability ({rr_std:.1f}ms) suggests irregular rhythm"
                )
            elif rr_std > 50:
                assessment["partial"] = True
                assessment["explanation"] = (
                    f"Moderate RR variability ({rr_std:.1f}ms) suggests some irregularity"
                )
            else:
                assessment["explanation"] = (
                    f"Low RR variability ({rr_std:.1f}ms) suggests regular rhythm"
                )

        elif "p wave" in criterion.description.lower():
            assessment["explanation"] = (
                "P wave morphology analysis requires detailed signal processing"
            )

        return assessment

    def _analyze_parameters(self, features: dict[str, Any]) -> dict[str, Any]:
        """Analyze ECG parameters against normal ranges"""

        analysis: dict[str, list[dict[str, Any]]] = {
            "normal_parameters": [],
            "abnormal_parameters": [],
            "borderline_parameters": [],
        }

        for param_name, normal_range in self.normal_ranges.items():
            if param_name in features:
                value = features[param_name]

                if normal_range.min_value <= value <= normal_range.max_value:
                    analysis["normal_parameters"].append(
                        {
                            "parameter": param_name,
                            "value": value,
                            "unit": normal_range.unit,
                            "normal_range": f"{normal_range.min_value}-{normal_range.max_value}",
                            "interpretation": "Within normal limits",
                        }
                    )
                else:
                    range_width = normal_range.max_value - normal_range.min_value
                    tolerance = range_width * 0.1

                    if (
                        normal_range.min_value - tolerance
                        <= value
                        <= normal_range.min_value
                        or normal_range.max_value
                        <= value
                        <= normal_range.max_value + tolerance
                    ):
                        analysis["borderline_parameters"].append(
                            {
                                "parameter": param_name,
                                "value": value,
                                "unit": normal_range.unit,
                                "normal_range": f"{normal_range.min_value}-{normal_range.max_value}",
                                "interpretation": "Borderline abnormal",
                            }
                        )
                    else:
                        if value < normal_range.min_value:
                            interpretation = f"Below normal (low {param_name})"
                        else:
                            interpretation = f"Above normal (high {param_name})"

                        analysis["abnormal_parameters"].append(
                            {
                                "parameter": param_name,
                                "value": value,
                                "unit": normal_range.unit,
                                "normal_range": f"{normal_range.min_value}-{normal_range.max_value}",
                                "interpretation": interpretation,
                            }
                        )

        return analysis

    def _explain_clinical_significance(
        self, diagnosis_code: str, features: dict[str, Any]
    ) -> str:
        """Explain the clinical significance of the diagnosis"""

        condition = get_condition_by_code(diagnosis_code)

        if not condition:
            return "Clinical significance cannot be determined for unknown condition"

        significance_map = {
            "STEMI": "ST-elevation myocardial infarction represents complete coronary artery occlusion requiring immediate reperfusion therapy. This is a medical emergency with high mortality risk if untreated.",
            "NSTEMI": "Non-ST elevation myocardial infarction indicates partial coronary artery occlusion with myocardial necrosis. Requires urgent cardiology evaluation and risk stratification.",
            "AFIB": "Atrial fibrillation increases stroke risk due to potential thrombus formation in the left atrium. Requires anticoagulation assessment and rate/rhythm control strategy.",
            "VTAC": "Ventricular tachycardia is a life-threatening arrhythmia that can degenerate into ventricular fibrillation. Requires immediate evaluation and treatment.",
            "LBBB": "Left bundle branch block may indicate underlying structural heart disease and can affect cardiac synchrony. May require pacemaker evaluation if symptomatic.",
            "RBBB": "Right bundle branch block is often benign in young patients but may indicate right heart strain or congenital heart disease in certain contexts.",
            "LVH": "Left ventricular hypertrophy indicates increased cardiac workload, often due to hypertension. Associated with increased cardiovascular risk.",
            "AVB3": "Complete heart block represents complete failure of AV conduction. Often requires permanent pacemaker implantation due to risk of asystole.",
        }

        base_significance = significance_map.get(
            diagnosis_code,
            f"Clinical significance of {condition.name} requires individual assessment based on patient context.",
        )

        if condition.clinical_urgency == "critical":
            urgency_context = " This condition requires immediate medical attention and emergency intervention."
        elif condition.clinical_urgency == "high":
            urgency_context = (
                " This condition requires urgent medical evaluation and treatment."
            )
        elif condition.clinical_urgency == "medium":
            urgency_context = " This condition requires timely medical evaluation and appropriate management."
        else:
            urgency_context = " This condition should be evaluated by a healthcare provider for appropriate management."

        return base_significance + urgency_context

    def _generate_differential_diagnosis(
        self, predictions: dict[str, float]
    ) -> list[dict[str, Any]]:
        """Generate differential diagnosis list"""

        sorted_predictions = sorted(
            predictions.items(), key=lambda x: x[1], reverse=True
        )

        differential = []

        for i, (diagnosis_code, confidence) in enumerate(
            sorted_predictions[:5]
        ):  # Top 5
            condition = get_condition_by_code(diagnosis_code)

            differential.append(
                {
                    "rank": i + 1,
                    "diagnosis_code": diagnosis_code,
                    "diagnosis_name": condition.name if condition else diagnosis_code,
                    "confidence": confidence,
                    "likelihood": self._interpret_confidence(confidence),
                    "category": condition.category if condition else "unknown",
                }
            )

        return differential

    def _generate_clinical_recommendations(
        self, diagnosis_code: str, features: dict[str, Any]
    ) -> list[str]:
        """Generate clinical recommendations based on diagnosis"""

        condition = get_condition_by_code(diagnosis_code)
        recommendations: list[str] = []

        if condition and condition.clinical_urgency == "critical":
            recommendations.extend(
                [
                    "URGENT: Immediate cardiology consultation required",
                    "Continuous cardiac monitoring",
                    "Prepare for emergency intervention if indicated",
                ]
            )
        elif condition and condition.clinical_urgency == "high":
            recommendations.extend(
                [
                    "Urgent cardiology evaluation within 24 hours",
                    "Serial ECGs to monitor progression",
                    "Consider telemetry monitoring",
                ]
            )

        diagnosis_recommendations = {
            "STEMI": [
                "Emergency cardiac catheterization (door-to-balloon <90 minutes)",
                "Dual antiplatelet therapy (aspirin + P2Y12 inhibitor)",
                "Anticoagulation with heparin",
                "Beta-blocker and ACE inhibitor when stable",
                "Serial troponin measurements",
            ],
            "NSTEMI": [
                "Risk stratification with TIMI or GRACE score",
                "Cardiac catheterization within 24-72 hours",
                "Dual antiplatelet therapy",
                "Anticoagulation based on bleeding risk",
                "Echocardiogram to assess wall motion",
            ],
            "AFIB": [
                "Assess CHA2DS2-VASc score for stroke risk",
                "Consider anticoagulation if score ≥2 (men) or ≥3 (women)",
                "Rate control with beta-blockers or calcium channel blockers",
                "Consider rhythm control in symptomatic patients",
                "Echocardiogram to assess atrial size and function",
            ],
            "VTAC": [
                "Immediate assessment of hemodynamic stability",
                "Cardioversion if hemodynamically unstable",
                "Antiarrhythmic therapy (amiodarone or lidocaine)",
                "Electrolyte correction (K+, Mg2+)",
                "Evaluate for underlying ischemia",
            ],
            "LBBB": [
                "Echocardiogram to assess left ventricular function",
                "Evaluate for underlying coronary artery disease",
                "Consider cardiac resynchronization therapy if EF <35%",
                "Monitor for progression to higher-degree blocks",
            ],
            "AVB3": [
                "Immediate pacemaker evaluation",
                "Temporary pacing if symptomatic or unstable",
                "Discontinue AV-blocking medications",
                "Monitor for escape rhythm adequacy",
            ],
        }

        specific_recs = diagnosis_recommendations.get(
            diagnosis_code,
            [
                "Follow standard clinical protocols for this condition",
                "Consider cardiology consultation if symptomatic",
                "Regular follow-up as clinically indicated",
            ],
        )

        recommendations.extend(specific_recs)

        return recommendations

    def _explain_anatomical_correlation(
        self,
        diagnosis_code: str,
        features: dict[str, Any],
        shap_explanation: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Explain anatomical correlation of findings"""

        correlation: dict[str, Any] = {
            "affected_territories": [],
            "lead_analysis": {},
            "vascular_correlation": {},
        }

        important_leads = []
        if shap_explanation and "lead_importance" in shap_explanation:
            lead_importance = shap_explanation["lead_importance"]
            sorted_leads = sorted(
                lead_importance.items(), key=lambda x: abs(x[1]), reverse=True
            )
            important_leads = [lead for lead, importance in sorted_leads[:3]]

        for territory, info in self.lead_territories.items():
            territory_leads = info["leads"]

            if any(lead in important_leads for lead in territory_leads):
                correlation["affected_territories"].append(
                    {
                        "territory": territory,
                        "leads": territory_leads,
                        "artery": info["artery"],
                        "anatomy": info["anatomy"],
                    }
                )

        for lead in important_leads:
            correlation["lead_analysis"][lead] = self._analyze_lead_significance(
                lead, diagnosis_code
            )

        return correlation

    def _analyze_lead_significance(self, lead: str, diagnosis_code: str) -> str:
        """Analyze the significance of findings in a specific lead"""

        lead_significance = {
            "II": "Lead II provides good visualization of inferior wall and P wave morphology",
            "III": "Lead III is sensitive for inferior wall changes, especially RCA territory",
            "aVF": "Lead aVF confirms inferior wall involvement when concordant with II and III",
            "V1": "Lead V1 shows septal wall and right ventricular activity",
            "V2": "Lead V2 represents septal to anterior wall transition",
            "V3": "Lead V3 shows anterior wall of left ventricle",
            "V4": "Lead V4 represents anterior wall and apex",
            "V5": "Lead V5 shows lateral wall of left ventricle",
            "V6": "Lead V6 represents lateral wall and left ventricular free wall",
        }

        base_significance = lead_significance.get(
            lead, f"Lead {lead} significance varies by clinical context"
        )

        if diagnosis_code == "STEMI":
            if lead in ["II", "III", "aVF"]:
                return (
                    base_significance
                    + " - Inferior STEMI pattern suggests RCA occlusion"
                )
            elif lead in ["V1", "V2", "V3", "V4"]:
                return (
                    base_significance
                    + " - Anterior STEMI pattern suggests LAD occlusion"
                )
            elif lead in ["I", "aVL", "V5", "V6"]:
                return (
                    base_significance
                    + " - Lateral STEMI pattern suggests LCX occlusion"
                )

        return base_significance

    def _assess_risk_stratification(
        self, diagnosis_code: str, features: dict[str, Any]
    ) -> dict[str, Any]:
        """Assess risk stratification for the diagnosis"""

        risk_assessment: dict[str, Any] = {
            "overall_risk": "moderate",
            "risk_factors": [],
            "protective_factors": [],
            "risk_scores": {},
        }

        age = features.get("patient_age", 50)
        if age > 75:
            risk_assessment["risk_factors"].append(f"Advanced age ({age} years)")
        elif age < 40:
            risk_assessment["protective_factors"].append(f"Young age ({age} years)")

        hr = features.get("heart_rate", 70)
        if diagnosis_code == "AFIB":
            if hr > 110:
                risk_assessment["risk_factors"].append("Rapid ventricular response")
                risk_assessment["overall_risk"] = "high"
            elif hr < 50:
                risk_assessment["risk_factors"].append("Slow ventricular response")

        qrs = features.get("qrs_duration", 100)
        if qrs > 150:
            risk_assessment["risk_factors"].append("Severely prolonged QRS duration")

        if diagnosis_code == "STEMI":
            risk_assessment["overall_risk"] = "critical"
            risk_assessment["risk_factors"].extend(
                [
                    "Acute coronary syndrome",
                    "Risk of cardiogenic shock",
                    "Risk of mechanical complications",
                ]
            )
        elif diagnosis_code == "VTAC":
            risk_assessment["overall_risk"] = "critical"
            risk_assessment["risk_factors"].append(
                "Risk of degeneration to ventricular fibrillation"
            )

        return risk_assessment

    def _recommend_follow_up(
        self, diagnosis_code: str, features: dict[str, Any]
    ) -> dict[str, Any]:
        """Recommend appropriate follow-up care"""

        follow_up: dict[str, list[str]] = {
            "immediate_actions": [],
            "short_term_follow_up": [],
            "long_term_monitoring": [],
            "specialist_referrals": [],
        }

        condition = get_condition_by_code(diagnosis_code)

        if condition and condition.clinical_urgency == "critical":
            follow_up["immediate_actions"].extend(
                [
                    "Emergency department evaluation",
                    "Continuous cardiac monitoring",
                    "Serial ECGs every 15-30 minutes",
                ]
            )
        elif condition and condition.clinical_urgency == "high":
            follow_up["immediate_actions"].extend(
                [
                    "Urgent cardiology consultation",
                    "Repeat ECG in 1-2 hours",
                    "Monitor vital signs closely",
                ]
            )

        follow_up_plans = {
            "STEMI": {
                "short_term_follow_up": [
                    "Daily ECGs during hospitalization",
                    "Echocardiogram within 24 hours",
                    "Cardiac rehabilitation referral",
                ],
                "long_term_monitoring": [
                    "Cardiology follow-up in 1-2 weeks",
                    "Repeat echocardiogram in 3 months",
                    "Annual stress testing",
                ],
                "specialist_referrals": [
                    "Interventional cardiology",
                    "Cardiac rehabilitation",
                ],
            },
            "AFIB": {
                "short_term_follow_up": [
                    "Anticoagulation assessment within 48 hours",
                    "Echocardiogram within 1 week",
                ],
                "long_term_monitoring": [
                    "Cardiology follow-up in 2-4 weeks",
                    "Annual echocardiogram",
                    "Periodic Holter monitoring",
                ],
                "specialist_referrals": [
                    "Cardiology",
                    "Electrophysiology if symptomatic",
                ],
            },
        }

        if diagnosis_code in follow_up_plans:
            plan = follow_up_plans[diagnosis_code]
            follow_up.update(plan)
        else:
            follow_up["short_term_follow_up"] = [
                "Repeat ECG in 24-48 hours if symptomatic"
            ]
            follow_up["long_term_monitoring"] = [
                "Cardiology follow-up as clinically indicated"
            ]

        return follow_up
