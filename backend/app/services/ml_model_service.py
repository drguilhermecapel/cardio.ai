"""
ML Model Service - Machine Learning model management and inference.
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from app.core.config import settings
from app.core.exceptions import MLModelException
from app.utils.memory_monitor import MemoryMonitor

logger = logging.getLogger(__name__)


class MLModelService:
    """ML Model Service for ECG analysis using ONNX Runtime."""

    def __init__(self) -> None:
        self.models: dict[str, ort.InferenceSession] = {}
        self.model_metadata: dict[str, dict[str, Any]] = {}
        self.memory_monitor = MemoryMonitor()
        self._load_models()

    def _load_models(self) -> None:
        """Load ML models from disk."""
        try:
            models_dir = Path(settings.MODELS_DIR)
            if not models_dir.exists():
                logger.warning(f"Models directory not found: {models_dir}")
                return

            ecg_model_path = models_dir / "ecg_classifier.onnx"
            if ecg_model_path.exists():
                self._load_model("ecg_classifier", str(ecg_model_path))

            rhythm_model_path = models_dir / "rhythm_detector.onnx"
            if rhythm_model_path.exists():
                self._load_model("rhythm_detector", str(rhythm_model_path))

            quality_model_path = models_dir / "quality_assessor.onnx"
            if quality_model_path.exists():
                self._load_model("quality_assessor", str(quality_model_path))

            logger.info(f"Loaded {len(self.models)} ML models")

        except Exception as e:
            logger.error(f"Failed to load models: {str(e)}")

    def _load_model(self, model_name: str, model_path: str) -> None:
        """Load a single ONNX model."""
        try:
            providers = ["CPUExecutionProvider"]

            if ort.get_device() == "GPU":
                providers.insert(0, "CUDAExecutionProvider")

            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )
            session_options.intra_op_num_threads = 4
            session_options.inter_op_num_threads = 2

            session = ort.InferenceSession(
                model_path, sess_options=session_options, providers=providers
            )

            self.models[model_name] = session

            input_meta = session.get_inputs()[0]
            output_meta = session.get_outputs()[0]

            self.model_metadata[model_name] = {
                "input_shape": input_meta.shape,
                "input_type": input_meta.type,
                "output_shape": output_meta.shape,
                "output_type": output_meta.type,
                "providers": session.get_providers(),
            }

            logger.info(
                f"Loaded model {model_name} with input_shape={input_meta.shape} providers={session.get_providers()}"
            )

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise MLModelException(
                f"Failed to load model {model_name}: {str(e)}"
            ) from e

    async def analyze_ecg(
        self,
        ecg_data: "np.ndarray[Any, np.dtype[np.float32]]",
        sample_rate: int,
        leads_names: list[str],
    ) -> dict[str, Any]:
        """Analyze ECG data using ML models."""
        try:
            start_time = time.time()

            processed_data = await self._preprocess_for_inference(
                ecg_data, sample_rate, leads_names
            )

            results = {
                "confidence": 0.0,
                "predictions": {},
                "interpretability": {},
                "rhythm": "Unknown",
                "events": [],
            }

            if "ecg_classifier" in self.models:
                classification_results = await self._run_classification(processed_data)
                predictions = classification_results["predictions"]
                if isinstance(predictions, dict):
                    results_predictions = results["predictions"]
                    if isinstance(results_predictions, dict):
                        results_predictions.update(predictions)
                results["confidence"] = classification_results["confidence"]

                if classification_results["confidence"] > 0.5:
                    interpretability = await self._generate_interpretability(
                        processed_data, classification_results
                    )
                    results["interpretability"] = interpretability

            if "rhythm_detector" in self.models:
                rhythm_results = await self._run_rhythm_detection(processed_data)
                results["rhythm"] = rhythm_results["rhythm"]
                events = rhythm_results.get("events", [])
                if isinstance(events, list):
                    results_events = results["events"]
                    if isinstance(results_events, list):
                        results_events.extend(events)

            if "quality_assessor" in self.models:
                quality_results = await self._run_quality_assessment(processed_data)
                results["quality_score"] = quality_results["score"]
                results["quality_issues"] = quality_results.get("issues", [])

            processing_time = time.time() - start_time
            results["processing_time_seconds"] = processing_time

            logger.info(
                f"ECG analysis completed with confidence={results['confidence']} rhythm={results['rhythm']} processing_time={processing_time}"
            )

            return results

        except Exception as e:
            logger.error(f"ECG analysis failed: {str(e)}")
            raise MLModelException(f"ECG analysis failed: {str(e)}") from e

    async def _preprocess_for_inference(
        self,
        ecg_data: "np.ndarray[Any, np.dtype[np.float32]]",
        sample_rate: int,
        leads_names: list[str],
    ) -> "np.ndarray[Any, np.dtype[np.float32]]":
        """Preprocess ECG data for model inference."""
        try:
            if ecg_data.shape[1] < 12:
                padded_data = np.zeros((ecg_data.shape[0], 12), dtype=np.float32)
                padded_data[:, : ecg_data.shape[1]] = ecg_data
                ecg_data = padded_data
            elif ecg_data.shape[1] > 12:
                ecg_data = ecg_data[:, :12].astype(np.float32)

            target_sample_rate = 500
            if sample_rate != target_sample_rate:
                from scipy import signal

                num_samples = int(ecg_data.shape[0] * target_sample_rate / sample_rate)
                ecg_data = signal.resample(ecg_data, num_samples, axis=0)

            ecg_data = (ecg_data - np.mean(ecg_data, axis=0)) / (
                np.std(ecg_data, axis=0) + 1e-8
            )

            target_length = 5000
            if ecg_data.shape[0] > target_length:
                ecg_data = ecg_data[:target_length, :]
            elif ecg_data.shape[0] < target_length:
                padded_data = np.zeros((target_length, 12), dtype=np.float32)
                padded_data[: ecg_data.shape[0], :] = ecg_data
                ecg_data = padded_data

            ecg_data = np.expand_dims(ecg_data.T, axis=0).astype(np.float32)

            return ecg_data

        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}")
            raise MLModelException(f"Preprocessing failed: {str(e)}") from e

    async def _run_classification(
        self, data: "np.ndarray[Any, np.dtype[np.float32]]"
    ) -> dict[str, Any]:
        """Run ECG classification model."""
        try:
            model = self.models["ecg_classifier"]
            input_name = model.get_inputs()[0].name

            outputs = model.run(None, {input_name: data})
            predictions = outputs[0][0]  # Remove batch dimension

            condition_names = [
                "normal",
                "atrial_fibrillation",
                "atrial_flutter",
                "supraventricular_tachycardia",
                "ventricular_tachycardia",
                "ventricular_fibrillation",
                "first_degree_block",
                "second_degree_block",
                "complete_heart_block",
                "left_bundle_branch_block",
                "right_bundle_branch_block",
                "premature_atrial_contraction",
                "premature_ventricular_contraction",
                "stemi",
                "nstemi",
            ]

            predictions_dict = {}
            for i, condition in enumerate(condition_names):
                if i < len(predictions):
                    predictions_dict[condition] = float(predictions[i])

            confidence = float(np.max(predictions))

            return {
                "predictions": predictions_dict,
                "confidence": confidence,
            }

        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            return {"predictions": {}, "confidence": 0.0}

    async def _run_rhythm_detection(
        self, data: "np.ndarray[Any, np.dtype[np.float32]]"
    ) -> dict[str, Any]:
        """Run rhythm detection model."""
        try:
            model = self.models["rhythm_detector"]
            input_name = model.get_inputs()[0].name

            outputs = model.run(None, {input_name: data})
            rhythm_probs = outputs[0][0]

            rhythm_types = [
                "Sinus Rhythm",
                "Atrial Fibrillation",
                "Atrial Flutter",
                "Supraventricular Tachycardia",
                "Ventricular Tachycardia",
                "Ventricular Fibrillation",
                "Asystole",
            ]

            rhythm_idx = np.argmax(rhythm_probs)
            rhythm = (
                rhythm_types[rhythm_idx]
                if rhythm_idx < len(rhythm_types)
                else "Unknown"
            )

            events = []
            if rhythm != "Sinus Rhythm" and rhythm_probs[rhythm_idx] > 0.7:
                events.append(
                    {
                        "label": f"{rhythm}_detected",
                        "time_ms": 0.0,
                        "confidence": float(rhythm_probs[rhythm_idx]),
                        "properties": {"rhythm_type": rhythm},
                    }
                )

            return {
                "rhythm": rhythm,
                "rhythm_confidence": float(rhythm_probs[rhythm_idx]),
                "events": events,
            }

        except Exception as e:
            logger.error(f"Rhythm detection failed: {str(e)}")
            return {"rhythm": "Unknown", "events": []}

    async def _run_quality_assessment(
        self, data: "np.ndarray[Any, np.dtype[np.float32]]"
    ) -> dict[str, Any]:
        """Run signal quality assessment model."""
        try:
            model = self.models["quality_assessor"]
            input_name = model.get_inputs()[0].name

            outputs = model.run(None, {input_name: data})
            quality_score = float(outputs[0][0][0])

            issues = []
            if quality_score < 0.5:
                issues.append("Poor signal quality")
            if quality_score < 0.3:
                issues.append("Significant noise detected")

            return {
                "score": quality_score,
                "issues": issues,
            }

        except Exception as e:
            logger.error(f"Quality assessment failed: {str(e)}")
            return {"score": 0.5, "issues": ["Quality assessment unavailable"]}

    async def _generate_interpretability(
        self,
        data: "np.ndarray[Any, np.dtype[np.float32]]",
        classification_results: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate interpretability maps using SHAP-based explanations."""
        try:
            from app.services.interpretability_service import InterpretabilityService

            interpretability_service = InterpretabilityService()

            signal = data[
                0
            ].T  # Remove batch dimension and transpose back to (time, leads)

            features = self._extract_features_for_interpretability(signal)

            explanation_result = (
                await interpretability_service.generate_comprehensive_explanation(
                    signal=signal,
                    features=features,
                    predictions=classification_results["predictions"],
                    model_output=classification_results,
                )
            )

            interpretability = {
                "attention_maps": explanation_result.attention_maps,
                "feature_importance": explanation_result.feature_importance,
                "explanation": explanation_result.clinical_explanation
                or "AI detected patterns consistent with the predicted condition",
                "shap_explanation": explanation_result.shap_explanation,
                "lime_explanation": explanation_result.lime_explanation,
                "clinical_explanation": explanation_result.clinical_explanation,
                "diagnostic_criteria": explanation_result.diagnostic_criteria,
                "risk_factors": explanation_result.risk_factors,
                "recommendations": explanation_result.recommendations,
            }

            return interpretability

        except Exception as e:
            logger.error(f"SHAP-based interpretability generation failed: {str(e)}")
            return {
                "explanation": "Interpretability analysis unavailable",
                "attention_maps": {},
                "feature_importance": {},
                "error": str(e),
            }

    def _extract_features_for_interpretability(
        self, signal: "np.ndarray[Any, np.dtype[np.float32]]"
    ) -> dict[str, Any]:
        """Extract basic features from ECG signal for interpretability analysis."""
        try:
            features: dict[str, Any] = {}

            features["signal_length"] = int(signal.shape[0])
            features["num_leads"] = int(signal.shape[1])

            if signal.shape[0] > 0:
                lead_ii = signal[:, 1] if signal.shape[1] > 1 else signal[:, 0]
                peaks = np.where(np.diff(np.sign(np.diff(lead_ii))) < 0)[0]
                if len(peaks) > 1:
                    rr_intervals = (
                        np.diff(peaks) / 500.0
                    )  # Assuming 500 Hz sampling rate
                    features["heart_rate"] = (
                        float(60.0 / np.mean(rr_intervals))
                        if len(rr_intervals) > 0
                        else 70.0
                    )
                    features["rr_mean"] = float(
                        np.mean(rr_intervals) * 1000
                    )  # Convert to ms
                    features["rr_std"] = float(np.std(rr_intervals) * 1000)
                else:
                    features["heart_rate"] = 70.0
                    features["rr_mean"] = 857.0  # ~70 bpm
                    features["rr_std"] = 50.0

            for i, lead_name in enumerate(
                [
                    "I",
                    "II",
                    "III",
                    "aVR",
                    "aVL",
                    "aVF",
                    "V1",
                    "V2",
                    "V3",
                    "V4",
                    "V5",
                    "V6",
                ]
            ):
                if i < signal.shape[1]:
                    lead_signal = signal[:, i]
                    features[f"{lead_name}_amplitude_max"] = float(np.max(lead_signal))
                    features[f"{lead_name}_amplitude_min"] = float(np.min(lead_signal))
                    features[f"{lead_name}_std"] = float(np.std(lead_signal))

            features["qrs_duration"] = (
                100.0  # Default values - would be calculated from actual signal
            )
            features["pr_interval"] = 160.0
            features["qt_interval"] = 400.0
            features["qtc"] = 420.0
            features["qrs_axis"] = 60.0

            features["st_elevation_max"] = 0.0
            features["st_depression_max"] = 0.0

            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return {
                "heart_rate": 70.0,
                "signal_length": int(signal.shape[0]) if signal.size > 0 else 0,
                "num_leads": int(signal.shape[1]) if signal.size > 0 else 0,
            }

    def get_model_info(self) -> dict[str, Any]:
        """Get information about loaded models."""
        return {
            "loaded_models": list(self.models.keys()),
            "model_metadata": self.model_metadata,
            "memory_usage": self.memory_monitor.get_memory_usage(),
        }

    def unload_model(self, model_name: str) -> bool:
        """Unload a model to free memory."""
        if model_name in self.models:
            del self.models[model_name]
            if model_name in self.model_metadata:
                del self.model_metadata[model_name]
            logger.info(f"Unloaded model {model_name}")
            return True
        return False
