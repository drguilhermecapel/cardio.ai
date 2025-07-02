"""
Multi-Pathology Service for comprehensive ECG analysis
Implements hierarchical classification approach
"""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple
import numpy as np
from enum import Enum
import copy

from app.core.constants import ClinicalUrgency, DiagnosisCode
from app.core.exceptions import MultiPathologyException
from app.ml.ml_model_service import MLModelService

logger = logging.getLogger(__name__)


class SCPCategory(Enum):
    """SCP ECG diagnostic categories"""

    NORMAL = "normal"
    ARRHYTHMIA = "arrhythmia"
    CONDUCTION = "conduction"
    ISCHEMIA = "ischemia"
    HYPERTROPHY = "hypertrophy"
    AXIS_DEVIATION = "axis_deviation"
    OTHER = "other"


class MultiPathologyService:
    """Service for multi-pathology ECG analysis"""

    # Complete SCP conditions mapping (71 conditions)
    SCP_CONDITIONS = {
        # Normal
        "NORM": {"index": 0, "category": SCPCategory.NORMAL, "severity": 0.0},
        "SR": {"index": 1, "category": SCPCategory.NORMAL, "severity": 0.0},
        "ISCA": {"index": 2, "category": SCPCategory.NORMAL, "severity": 0.1},
        "ISCI": {"index": 3, "category": SCPCategory.NORMAL, "severity": 0.2},
        "ASMI": {"index": 48, "category": SCPCategory.NORMAL, "severity": 0.3},
        # Arrhythmias
        "AFIB": {"index": 9, "category": SCPCategory.ARRHYTHMIA, "severity": 0.9},
        "AFLT": {"index": 10, "category": SCPCategory.ARRHYTHMIA, "severity": 0.8},
        "PSVT": {"index": 11, "category": SCPCategory.ARRHYTHMIA, "severity": 0.7},
        "SVTAC": {"index": 12, "category": SCPCategory.ARRHYTHMIA, "severity": 0.7},
        "AT": {"index": 13, "category": SCPCategory.ARRHYTHMIA, "severity": 0.6},
        "AVNRT": {"index": 14, "category": SCPCategory.ARRHYTHMIA, "severity": 0.6},
        "AVRT": {"index": 15, "category": SCPCategory.ARRHYTHMIA, "severity": 0.6},
        "SAAWR": {"index": 16, "category": SCPCategory.ARRHYTHMIA, "severity": 0.5},
        "STACH": {"index": 17, "category": SCPCategory.ARRHYTHMIA, "severity": 0.6},
        "SARRH": {"index": 18, "category": SCPCategory.ARRHYTHMIA, "severity": 0.4},
        "SBRAD": {"index": 19, "category": SCPCategory.ARRHYTHMIA, "severity": 0.5},
        "PACE": {"index": 20, "category": SCPCategory.ARRHYTHMIA, "severity": 0.3},
        "TRIGU": {"index": 21, "category": SCPCategory.ARRHYTHMIA, "severity": 0.4},
        "BIGU": {"index": 22, "category": SCPCategory.ARRHYTHMIA, "severity": 0.4},
        "VEB": {"index": 23, "category": SCPCategory.ARRHYTHMIA, "severity": 0.5},
        "SVEB": {"index": 24, "category": SCPCategory.ARRHYTHMIA, "severity": 0.4},
        # Conduction disturbances
        "IAVB": {"index": 4, "category": SCPCategory.CONDUCTION, "severity": 0.6},
        "AVB2": {"index": 5, "category": SCPCategory.CONDUCTION, "severity": 0.7},
        "AVB3": {"index": 6, "category": SCPCategory.CONDUCTION, "severity": 0.9},
        "LAFB": {"index": 7, "category": SCPCategory.CONDUCTION, "severity": 0.4},
        "LPFB": {"index": 8, "category": SCPCategory.CONDUCTION, "severity": 0.4},
        "LBBB": {"index": 25, "category": SCPCategory.CONDUCTION, "severity": 0.6},
        "RBBB": {"index": 26, "category": SCPCategory.CONDUCTION, "severity": 0.5},
        "IRBBB": {"index": 27, "category": SCPCategory.CONDUCTION, "severity": 0.4},
        "WPW": {"index": 28, "category": SCPCategory.CONDUCTION, "severity": 0.7},
        # Ischemia/Infarction
        "MI": {"index": 29, "category": SCPCategory.ISCHEMIA, "severity": 1.0},
        "AMI": {"index": 30, "category": SCPCategory.ISCHEMIA, "severity": 1.0},
        "ALMI": {"index": 31, "category": SCPCategory.ISCHEMIA, "severity": 0.9},
        "ILMI": {"index": 32, "category": SCPCategory.ISCHEMIA, "severity": 0.9},
        "IPLMI": {"index": 33, "category": SCPCategory.ISCHEMIA, "severity": 0.8},
        "IPMI": {"index": 34, "category": SCPCategory.ISCHEMIA, "severity": 0.8},
        "PMI": {"index": 35, "category": SCPCategory.ISCHEMIA, "severity": 0.8},
        "IMI": {"index": 36, "category": SCPCategory.ISCHEMIA, "severity": 0.9},
        "INJAL": {"index": 37, "category": SCPCategory.ISCHEMIA, "severity": 0.8},
        "INJIL": {"index": 38, "category": SCPCategory.ISCHEMIA, "severity": 0.8},
        "INJIN": {"index": 39, "category": SCPCategory.ISCHEMIA, "severity": 0.8},
        "INJLA": {"index": 40, "category": SCPCategory.ISCHEMIA, "severity": 0.8},
        "INJAS": {"index": 41, "category": SCPCategory.ISCHEMIA, "severity": 0.8},
        # Hypertrophy
        "LVH": {"index": 42, "category": SCPCategory.HYPERTROPHY, "severity": 0.6},
        "RVH": {"index": 43, "category": SCPCategory.HYPERTROPHY, "severity": 0.6},
        "LAH": {"index": 44, "category": SCPCategory.HYPERTROPHY, "severity": 0.5},
        "RAH": {"index": 45, "category": SCPCategory.HYPERTROPHY, "severity": 0.5},
        # Axis deviations
        "LAD": {"index": 46, "category": SCPCategory.AXIS_DEVIATION, "severity": 0.3},
        "RAD": {"index": 47, "category": SCPCategory.AXIS_DEVIATION, "severity": 0.3},
        # Other conditions
        "NDT": {"index": 49, "category": SCPCategory.OTHER, "severity": 0.3},
        "NST": {"index": 50, "category": SCPCategory.OTHER, "severity": 0.4},
        "DIG": {"index": 51, "category": SCPCategory.OTHER, "severity": 0.2},
        "LNGQT": {"index": 52, "category": SCPCategory.OTHER, "severity": 0.7},
        "ABQRS": {"index": 53, "category": SCPCategory.OTHER, "severity": 0.5},
        "PVC": {"index": 54, "category": SCPCategory.OTHER, "severity": 0.5},
        "STD": {"index": 55, "category": SCPCategory.OTHER, "severity": 0.6},
        "STE": {"index": 56, "category": SCPCategory.OTHER, "severity": 0.7},
        "TAB": {"index": 57, "category": SCPCategory.OTHER, "severity": 0.4},
        "INVT": {"index": 58, "category": SCPCategory.OTHER, "severity": 0.5},
        "LPR": {"index": 59, "category": SCPCategory.OTHER, "severity": 0.3},
        "LQT": {"index": 60, "category": SCPCategory.OTHER, "severity": 0.6},
        "QAB": {"index": 61, "category": SCPCategory.OTHER, "severity": 0.5},
        "QWAVE": {"index": 62, "category": SCPCategory.OTHER, "severity": 0.6},
        "SQT": {"index": 63, "category": SCPCategory.OTHER, "severity": 0.4},
        "TAB_": {"index": 64, "category": SCPCategory.OTHER, "severity": 0.4},
        "TIN": {"index": 65, "category": SCPCategory.OTHER, "severity": 0.4},
        "ISC_": {"index": 66, "category": SCPCategory.OTHER, "severity": 0.5},
        "ISCAL": {"index": 67, "category": SCPCategory.OTHER, "severity": 0.5},
        "ISCIN": {"index": 68, "category": SCPCategory.OTHER, "severity": 0.5},
        "ISCLA": {"index": 69, "category": SCPCategory.OTHER, "severity": 0.5},
        "ISCAS": {"index": 70, "category": SCPCategory.OTHER, "severity": 0.5},
    }

    def __init__(self, ml_service: Optional[MLModelService] = None):
        self.ml_service = ml_service
        self.scp_conditions = copy.deepcopy(self.SCP_CONDITIONS)
        self.condition_categories = self._build_category_mapping()
        self.category_thresholds = {
            SCPCategory.NORMAL: 0.99,  # High threshold for normal
            SCPCategory.ARRHYTHMIA: 0.5,
            SCPCategory.CONDUCTION: 0.5,
            SCPCategory.ISCHEMIA: 0.6,
            SCPCategory.HYPERTROPHY: 0.5,
            SCPCategory.AXIS_DEVIATION: 0.4,
            SCPCategory.OTHER: 0.4,
        }

    def _build_category_mapping(self) -> Dict[SCPCategory, List[str]]:
        """Build mapping from categories to conditions"""
        mapping = {}
        for condition, info in self.SCP_CONDITIONS.items():
            category = info["category"]
            if category not in mapping:
                mapping[category] = []
            mapping[category].append(condition)
        return mapping

    async def analyze_hierarchical(
        self,
        signal: np.ndarray,
        features: Dict[str, float],
        preprocessing_quality: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Perform hierarchical multi-pathology analysis"""
        try:
            # Level 1: Normal vs Abnormal
            level1_result = await self._classify_normal_vs_abnormal(signal, features)
            if level1_result["is_normal"]:
                return {
                    "diagnosis": "NORMAL",
                    "confidence": level1_result["confidence"],
                    "details": level1_result,
                    "detected_conditions": [],
                    "clinical_urgency": ClinicalUrgency.LOW,
                }

            # Level 2: Category classification
            level2_result = await self._classify_categories(signal, features)

            # Level 3: Specific condition detection
            level3_result = await self._detect_specific_conditions(
                signal, features, level2_result["predicted_categories"]
            )

            # Assess clinical urgency
            urgency = self._assess_clinical_urgency(
                level3_result["detected_conditions"]
            )

            # Get primary diagnosis
            primary = self._get_primary_diagnosis(level3_result["detected_conditions"])

            return {
                "diagnosis": primary["condition"] if primary else "ABNORMAL",
                "confidence": (
                    primary["confidence"] if primary else level2_result["confidence"]
                ),
                "details": {
                    "level1": level1_result,
                    "level2": level2_result,
                    "level3": level3_result,
                },
                "detected_conditions": level3_result["detected_conditions"],
                "clinical_urgency": urgency,
            }

        except Exception as e:
            logger.error(f"Error in hierarchical analysis: {str(e)}", exc_info=True)
            raise MultiPathologyException(f"Hierarchical analysis failed: {str(e)}")

    async def _classify_normal_vs_abnormal(
        self, signal: np.ndarray, features: Dict[str, float]
    ) -> Dict[str, Any]:
        """Level 1: Classify normal vs abnormal"""
        try:
            abnormal_indicators = []

            # Check heart rate
            hr = features.get("heart_rate", 75)
            if hr < 50 or hr > 120:
                abnormal_indicators.append("abnormal_heart_rate")

            # Check intervals
            pr = features.get("pr_interval", 160)
            if pr < 120 or pr > 220:
                abnormal_indicators.append("abnormal_pr_interval")

            qrs = features.get("qrs_duration", 90)
            if qrs > 120:
                abnormal_indicators.append("wide_qrs")

            qt = features.get("qt_interval", 400)
            qtc = features.get("qtc", qt)
            if qtc > 470:
                abnormal_indicators.append("prolonged_qtc")

            # Calculate probability
            is_normal = len(abnormal_indicators) == 0
            confidence = 0.98 if is_normal else 0.02

            # Calculate NPV score
            npv_score = 0.99 if is_normal else 0.01

            return {
                "is_normal": is_normal,
                "confidence": confidence,
                "abnormal_probability": 1 - confidence,
                "abnormal_indicators": abnormal_indicators,
                "npv_score": npv_score,
            }

        except Exception as e:
            logger.error(
                f"Error in normal/abnormal classification: {str(e)}", exc_info=True
            )
            return {
                "is_normal": True,
                "confidence": 0.5,
                "abnormal_probability": 0.5,
                "abnormal_indicators": [],
                "npv_score": 0.5,
            }

    async def _classify_categories(
        self, signal: np.ndarray, features: Dict[str, float]
    ) -> Dict[str, Any]:
        """Level 2: Classify into diagnostic categories"""
        try:
            category_probs = {}

            # Simple heuristic-based category classification
            hr = features.get("heart_rate", 75)
            pr = features.get("pr_interval", 160)
            qrs = features.get("qrs_duration", 90)

            # Arrhythmia probability
            if hr < 50 or hr > 150 or features.get("rr_std", 0) > 50:
                category_probs[SCPCategory.ARRHYTHMIA] = 0.7
            else:
                category_probs[SCPCategory.ARRHYTHMIA] = 0.1

            # Conduction probability
            if pr > 200 or qrs > 120:
                category_probs[SCPCategory.CONDUCTION] = 0.6
            else:
                category_probs[SCPCategory.CONDUCTION] = 0.1

            # Ischemia probability
            st_elevation = features.get("st_elevation", 0)
            if st_elevation > 1:
                category_probs[SCPCategory.ISCHEMIA] = 0.8
            else:
                category_probs[SCPCategory.ISCHEMIA] = 0.1

            # Other categories
            category_probs[SCPCategory.HYPERTROPHY] = 0.1
            category_probs[SCPCategory.AXIS_DEVIATION] = 0.1
            category_probs[SCPCategory.OTHER] = 0.1

            # Get predicted categories
            predicted_categories = [
                cat
                for cat, prob in category_probs.items()
                if cat in self.category_thresholds
                and prob > self.category_thresholds[cat]
            ]

            # Get primary category
            primary_category = max(category_probs.items(), key=lambda x: x[1])[0]

            return {
                "category_probabilities": category_probs,
                "predicted_categories": predicted_categories,
                "predicted_category": primary_category,
                "confidence": max(category_probs.values()),
            }

        except Exception as e:
            logger.error(f"Error in category classification: {str(e)}", exc_info=True)
            return {
                "category_probabilities": {},
                "predicted_categories": [],
                "predicted_category": SCPCategory.OTHER,
                "confidence": 0.5,
            }

    async def _detect_specific_conditions(
        self,
        signal: np.ndarray,
        features: Dict[str, float],
        categories: List[SCPCategory],
    ) -> Dict[str, Any]:
        """Level 3: Detect specific conditions within categories"""
        try:
            # Garantir que todos os itens são SCPCategory (enum)
            categories = [
                SCPCategory(cat) if not isinstance(cat, SCPCategory) else cat
                for cat in categories
            ]

            detected_conditions = []
            all_conditions = {}

            # Detect conditions based on features and categories
            hr = features.get("heart_rate", 75)
            pr = features.get("pr_interval", 160)
            qrs = features.get("qrs_duration", 90)

            # Arrhythmias
            if SCPCategory.ARRHYTHMIA in categories:
                if hr > 150 and features.get("rr_std", 0) > 100:
                    detected_conditions.append(
                        {
                            "condition": "AFIB",
                            "confidence": 0.8,
                            "severity": self.scp_conditions["AFIB"]["severity"],
                        }
                    )
                    all_conditions["AFIB"] = 0.8
                elif hr < 50:
                    detected_conditions.append(
                        {
                            "condition": "SBRAD",
                            "confidence": 0.7,
                            "severity": self.scp_conditions["SBRAD"]["severity"],
                        }
                    )
                    all_conditions["SBRAD"] = 0.7

            # Conduction disturbances
            if SCPCategory.CONDUCTION in categories:
                if pr > 200:
                    detected_conditions.append(
                        {
                            "condition": "IAVB",
                            "confidence": 0.7,
                            "severity": self.scp_conditions["IAVB"]["severity"],
                        }
                    )
                    all_conditions["IAVB"] = 0.7
                if qrs > 120:
                    detected_conditions.append(
                        {
                            "condition": "LBBB",
                            "confidence": 0.6,
                            "severity": self.scp_conditions["LBBB"]["severity"],
                        }
                    )
                    all_conditions["LBBB"] = 0.6

            # Sort by severity and confidence
            detected_conditions.sort(
                key=lambda x: (x["severity"], x["confidence"]), reverse=True
            )

            # Calculate confidence scores
            confidence_scores = {
                cond["condition"]: cond["confidence"] for cond in detected_conditions
            }

            return {
                "detected_conditions": detected_conditions,
                "all_conditions": all_conditions,
                "filtered_conditions": detected_conditions,
                "confidence": (
                    max([c["confidence"] for c in detected_conditions])
                    if detected_conditions
                    else 0.5
                ),
                "confidence_scores": confidence_scores,
            }

        except Exception as e:
            logger.error(
                f"Error detecting specific conditions: {str(e)}", exc_info=True
            )
            return {
                "detected_conditions": [],
                "all_conditions": {},
                "filtered_conditions": [],
                "confidence": 0.5,
                "confidence_scores": {},
            }

    def _assess_clinical_urgency(
        self, conditions: List[Dict[str, Any]]
    ) -> ClinicalUrgency:
        """Assess clinical urgency based on detected conditions"""
        if not conditions:
            return ClinicalUrgency.LOW

        # Get highest severity
        max_severity = max(cond["severity"] for cond in conditions)

        if max_severity >= 0.9:
            return ClinicalUrgency.CRITICAL
        elif max_severity >= 0.7:
            return ClinicalUrgency.HIGH
        elif max_severity >= 0.5:
            return ClinicalUrgency.MEDIUM
        else:
            return ClinicalUrgency.LOW

    def _get_primary_diagnosis(
        self, conditions: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Get primary diagnosis from detected conditions"""
        if not conditions:
            return None

        # Return condition with highest severity and confidence
        return max(conditions, key=lambda x: (x["severity"], x["confidence"]))

    async def analyze_pathologies(
        self, signal: np.ndarray, features: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze signal for multiple pathologies (backward compatibility)"""
        preprocessing_quality = {"overall_quality": 0.8}
        return await self.analyze_hierarchical(signal, features, preprocessing_quality)
