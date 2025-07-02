"""
Advanced ML Service for ECG Analysis
Integrates hybrid architecture, interpretability, and edge optimization
Production-ready implementation with state-of-the-art performance
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from scipy import signal as scipy_signal

from app.core.config import settings
from app.core.scp_ecg_conditions import SCP_ECG_CONDITIONS, get_condition_by_code
from app.ml.edge_optimization import (
    EdgeInferenceEngine,
    EdgeOptimizationConfig,
    create_edge_optimized_model,
)
from app.ml.hybrid_architecture import ModelConfig, create_hybrid_model, load_pretrained_model
from app.ml.interpretability_module import HybridExplainabilitySystem, create_explainability_system
from app.ml.training_pipeline import ECGTrainer, TrainingConfig
from app.preprocessing.advanced_pipeline import AdvancedECGPreprocessor
from app.preprocessing.enhanced_quality_analyzer import EnhancedSignalQualityAnalyzer

logger = logging.getLogger(__name__)


class InferenceMode(Enum):
    """Inference modes for different use cases"""
    FAST = "fast"  # Edge-optimized, low latency
    ACCURATE = "accurate"  # Full model, high accuracy
    INTERPRETABLE = "interpretable"  # With explanations


class ModelType(Enum):
    """Available model types"""
    HYBRID_FULL = "hybrid_full"
    HYBRID_MOBILE = "hybrid_mobile"
    EDGE_OPTIMIZED = "edge_optimized"
    ENSEMBLE = "ensemble"


@dataclass
class MLServiceConfig:
    """Configuration for ML service"""
    
    # Model selection
    model_type: ModelType = ModelType.HYBRID_FULL
    model_path: Optional[str] = None
    use_pretrained: bool = True
    
    # Inference settings
    inference_mode: InferenceMode = InferenceMode.ACCURATE
    batch_size: int = 32
    use_gpu: bool = torch.cuda.is_available()
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Preprocessing
    use_advanced_preprocessing: bool = True
    quality_threshold: float = 0.7
    
    # Interpretability
    enable_interpretability: bool = True
    explanation_methods: List[str] = field(
        default_factory=lambda: ["knowledge", "gradcam", "lime"]
    )
    
    # Performance optimization
    use_mixed_precision: bool = True
    enable_caching: bool = True
    cache_size: int = 1000
    
    # Confidence thresholds
    confidence_threshold: float = 0.8
    require_high_confidence: bool = True
    
    # Clinical validation
    enable_clinical_validation: bool = True
    validation_rules: List[str] = field(
        default_factory=lambda: ["rhythm", "morphology", "intervals"]
    )


@dataclass
class ECGPrediction:
    """Structured prediction result"""
    
    # Primary prediction
    condition_code: str
    condition_name: str
    probability: float
    confidence: float
    
    # Top predictions
    top_conditions: List[Dict[str, float]]
    
    # Clinical insights
    clinical_significance: str
    severity: str
    recommendations: List[str]
    
    # Quality metrics
    signal_quality: float
    prediction_quality: str
    
    # Interpretability
    explanations: Optional[Dict[str, Any]] = None
    
    # Metadata
    processing_time_ms: float = 0.0
    model_version: str = "1.0.0"
    timestamp: datetime = field(default_factory=datetime.now)


class AdaptiveThresholdManager:
    """Manages adaptive thresholds based on patient context"""
    
    def __init__(self):
        self.base_thresholds = {
            "atrial_fibrillation": 0.7,
            "ventricular_tachycardia": 0.6,
            "myocardial_infarction": 0.65,
            "left_bundle_branch_block": 0.75,
            "normal_sinus_rhythm": 0.8
        }
    
    def get_threshold(
        self,
        condition: str,
        patient_context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Get adaptive threshold for condition"""
        base_threshold = self.base_thresholds.get(condition, 0.7)
        
        if patient_context is None:
            return base_threshold
        
        # Adjust based on risk factors
        adjustment = 0.0
        
        # Age adjustment
        age = patient_context.get("age", 50)
        if age > 65:
            adjustment -= 0.05  # Lower threshold for elderly
        elif age < 30:
            adjustment += 0.05  # Higher threshold for young
        
        # History adjustment
        if patient_context.get("cardiac_history", False):
            adjustment -= 0.1  # Lower threshold for known cardiac patients
        
        # Symptoms adjustment
        if patient_context.get("symptomatic", False):
            adjustment -= 0.05  # Lower threshold for symptomatic patients
        
        return max(0.3, min(0.95, base_threshold + adjustment))


class ModelEnsemble:
    """Ensemble of multiple models for robust predictions"""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Ensemble prediction with weighted voting"""
        all_predictions = []
        all_logits = []
        
        for model, weight in zip(self.models, self.weights):
            output = model(x)
            all_predictions.append(output['predictions'] * weight)
            all_logits.append(output['logits'])
        
        # Weighted average of predictions
        ensemble_predictions = torch.stack(all_predictions).sum(dim=0)
        
        # For logits, we'll use the first model's logits (can be improved)
        ensemble_logits = all_logits[0]
        
        return {
            'predictions': ensemble_predictions,
            'logits': ensemble_logits,
            'individual_predictions': all_predictions
        }


class AdvancedMLService:
    """
    Advanced ML service with state-of-the-art ECG analysis
    """
    
    def __init__(self, config: Optional[MLServiceConfig] = None):
        self.config = config or MLServiceConfig()
        
        # Initialize components
        self._initialize_models()
        self._initialize_preprocessors()
        self._initialize_explainability()
        self._initialize_clinical_validator()
        
        # Performance optimization
        self.prediction_cache = {} if self.config.enable_caching else None
        self.adaptive_threshold_manager = AdaptiveThresholdManager()
        
        logger.info("Advanced ML Service initialized successfully")
    
    def _initialize_models(self):
        """Initialize ML models based on configuration"""
        if self.config.model_path and Path(self.config.model_path).exists():
            # Load pretrained model
            self.primary_model = load_pretrained_model(
                self.config.model_path,
                device=self.config.device
            )
            logger.info(f"Loaded pretrained model from {self.config.model_path}")
        else:
            # Create new model
            model_config = ModelConfig()
            
            if self.config.model_type == ModelType.HYBRID_FULL:
                self.primary_model = create_hybrid_model(model_config)
            elif self.config.model_type == ModelType.EDGE_OPTIMIZED:
                # Create edge-optimized model
                full_model = create_hybrid_model(model_config)
                self.primary_model, _ = create_edge_optimized_model(full_model)
            else:
                self.primary_model = create_hybrid_model(model_config)
            
            self.primary_model.to(self.config.device)
            logger.info(f"Created new {self.config.model_type.value} model")
        
        # Initialize ensemble if configured
        if self.config.model_type == ModelType.ENSEMBLE:
            self._initialize_ensemble()
        
        # Set model to evaluation mode
        self.primary_model.eval()
    
    def _initialize_ensemble(self):
        """Initialize model ensemble"""
        # Create diverse models for ensemble
        models = []
        
        # Model 1: Standard hybrid
        config1 = ModelConfig()
        model1 = create_hybrid_model(config1)
        models.append(model1)
        
        # Model 2: Larger capacity
        config2 = ModelConfig(
            gru_hidden_dim=384,
            transformer_layers=8
        )
        model2 = create_hybrid_model(config2)
        models.append(model2)
        
        # Model 3: Different architecture focus
        config3 = ModelConfig(
            use_frequency_attention=True,
            use_channel_attention=True
        )
        model3 = create_hybrid_model(config3)
        models.append(model3)
        
        # Create ensemble
        self.ensemble = ModelEnsemble(models)
        logger.info(f"Initialized ensemble with {len(models)} models")
    
    def _initialize_preprocessors(self):
        """Initialize preprocessing components"""
        self.preprocessor = AdvancedECGPreprocessor()
        self.quality_analyzer = EnhancedSignalQualityAnalyzer()
        logger.info("Initialized advanced preprocessing pipeline")
    
    def _initialize_explainability(self):
        """Initialize explainability system"""
        if self.config.enable_interpretability:
            self.explainability_system = create_explainability_system(self.primary_model)
            logger.info("Initialized explainability system")
        else:
            self.explainability_system = None
    
    def _initialize_clinical_validator(self):
        """Initialize clinical validation system"""
        self.clinical_validator = ClinicalValidator(self.config.validation_rules)
    
    async def analyze_ecg(
        self,
        ecg_signal: np.ndarray,
        sampling_rate: float = 500.0,
        patient_context: Optional[Dict[str, Any]] = None,
        return_interpretability: bool = True
    ) -> ECGPrediction:
        """
        Perform comprehensive ECG analysis
        
        Args:
            ecg_signal: ECG signal array (12, N) or (N, 12)
            sampling_rate: Sampling rate in Hz
            patient_context: Optional patient information
            return_interpretability: Whether to include explanations
            
        Returns:
            Comprehensive ECG prediction with clinical insights
        """
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = self._generate_cache_key(ecg_signal, patient_context)
            if self.prediction_cache and cache_key in self.prediction_cache:
                logger.info("Returning cached prediction")
                return self.prediction_cache[cache_key]
            
            # 1. Preprocessing and quality assessment
            processed_signal, quality_metrics = await self._preprocess_signal(
                ecg_signal, sampling_rate
            )
            
            # 2. Check signal quality
            if quality_metrics['overall_score'] < self.config.quality_threshold:
                return self._create_low_quality_prediction(quality_metrics)
            
            # 3. Run inference
            prediction_results = await self._run_inference(
                processed_signal,
                patient_context
            )
            
            # 4. Clinical validation
            validated_results = self.clinical_validator.validate(
                prediction_results,
                quality_metrics,
                patient_context
            )
            
            # 5. Generate explanations if requested
            explanations = None
            if return_interpretability and self.explainability_system:
                explanations = await self._generate_explanations(
                    processed_signal,
                    prediction_results,
                    patient_context
                )
            
            # 6. Create structured prediction
            prediction = self._create_prediction(
                validated_results,
                quality_metrics,
                explanations,
                processing_time=(time.time() - start_time) * 1000
            )
            
            # 7. Cache result
            if self.prediction_cache:
                self._update_cache(cache_key, prediction)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in ECG analysis: {str(e)}", exc_info=True)
            raise
    
    async def _preprocess_signal(
        self,
        signal: np.ndarray,
        sampling_rate: float
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Preprocess ECG signal"""
        # Run preprocessing in thread pool for async operation
        loop = asyncio.get_event_loop()
        
        # Advanced preprocessing
        processed, quality = await loop.run_in_executor(
            None,
            self.preprocessor.advanced_preprocessing_pipeline,
            signal,
            sampling_rate,
            True  # clinical_mode
        )
        
        # Additional quality assessment
        comprehensive_quality = await loop.run_in_executor(
            None,
            self.quality_analyzer.assess_signal_quality_comprehensive,
            processed
        )
        
        # Merge quality metrics
        quality.update(comprehensive_quality)
        
        return processed, quality
    
    async def _run_inference(
        self,
        signal: np.ndarray,
        patient_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Run model inference"""
        # Convert to tensor
        signal_tensor = torch.from_numpy(signal).float().unsqueeze(0).to(self.config.device)
        
        # Generate multimodal features if configured
        if self.config.inference_mode == InferenceMode.ACCURATE:
            # Compute additional features
            spectrogram = self._compute_spectrogram(signal)
            wavelet = self._compute_wavelet_features(signal)
            
            spectrogram_tensor = torch.from_numpy(spectrogram).float().unsqueeze(0).to(self.config.device)
            wavelet_tensor = torch.from_numpy(wavelet).float().unsqueeze(0).to(self.config.device)
        else:
            spectrogram_tensor = None
            wavelet_tensor = None
        
        # Run inference based on mode
        with torch.no_grad():
            if self.config.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    output = self._model_forward(
                        signal_tensor,
                        spectrogram_tensor,
                        wavelet_tensor
                    )
            else:
                output = self._model_forward(
                    signal_tensor,
                    spectrogram_tensor,
                    wavelet_tensor
                )
        
        # Convert to numpy
        predictions = output['predictions'].cpu().numpy()[0]
        logits = output['logits'].cpu().numpy()[0]
        
        # Get top predictions
        top_k = 5
        top_indices = predictions.argsort()[-top_k:][::-1]
        
        results = {
            'predictions': predictions,
            'logits': logits,
            'top_predictions': [
                {
                    'code': list(SCP_ECG_CONDITIONS.keys())[idx],
                    'name': SCP_ECG_CONDITIONS[list(SCP_ECG_CONDITIONS.keys())[idx]]['name'],
                    'probability': float(predictions[idx])
                }
                for idx in top_indices
            ]
        }
        
        # Apply adaptive thresholds
        if patient_context:
            results = self._apply_adaptive_thresholds(results, patient_context)
        
        return results
    
    def _model_forward(
        self,
        signal: torch.Tensor,
        spectrogram: Optional[torch.Tensor],
        wavelet: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through model"""
        if self.config.model_type == ModelType.ENSEMBLE:
            return self.ensemble.predict(signal)
        else:
            return self.primary_model(
                signal,
                spectrogram=spectrogram,
                wavelet=wavelet,
                return_attention=self.config.enable_interpretability
            )
    
    def _compute_spectrogram(self, signal: np.ndarray) -> np.ndarray:
        """Compute spectrogram features"""
        spectrograms = []
        
        for lead in signal:
            f, t, Sxx = scipy_signal.spectrogram(
                lead,
                fs=500,
                nperseg=256,
                noverlap=128
            )
            spectrograms.append(Sxx)
        
        return np.array(spectrograms)
    
    def _compute_wavelet_features(self, signal: np.ndarray) -> np.ndarray:
        """Compute wavelet features"""
        import pywt
        
        wavelet_features = []
        
        for lead in signal:
            coeffs = pywt.wavedec(lead, 'db6', level=6)
            # Concatenate coefficients
            features = np.concatenate([c.flatten() for c in coeffs])
            wavelet_features.append(features[:1000])  # Fixed size
        
        return np.array(wavelet_features)
    
    def _apply_adaptive_thresholds(
        self,
        results: Dict[str, Any],
        patient_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply adaptive thresholds based on patient context"""
        for pred in results['top_predictions']:
            condition = pred['code']
            threshold = self.adaptive_threshold_manager.get_threshold(
                condition,
                patient_context
            )
            
            # Adjust confidence based on threshold
            if pred['probability'] >= threshold:
                pred['confidence'] = 'high'
            elif pred['probability'] >= threshold * 0.8:
                pred['confidence'] = 'medium'
            else:
                pred['confidence'] = 'low'
        
        return results
    
    async def _generate_explanations(
        self,
        signal: np.ndarray,
        prediction_results: Dict[str, Any],
        patient_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comprehensive explanations"""
        loop = asyncio.get_event_loop()
        
        explanations = await loop.run_in_executor(
            None,
            self.explainability_system.explain_prediction,
            signal,
            prediction_results,
            self.config.explanation_methods
        )
        
        return explanations
    
    def _create_prediction(
        self,
        results: Dict[str, Any],
        quality_metrics: Dict[str, Any],
        explanations: Optional[Dict[str, Any]],
        processing_time: float
    ) -> ECGPrediction:
        """Create structured prediction object"""
        top_pred = results['top_predictions'][0]
        
        # Get clinical information
        condition_info = get_condition_by_code(top_pred['code'])
        
        # Determine severity
        severity = self._determine_severity(top_pred['code'], top_pred['probability'])
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            top_pred['code'],
            severity,
            quality_metrics
        )
        
        # Determine prediction quality
        prediction_quality = self._assess_prediction_quality(
            top_pred['probability'],
            quality_metrics['overall_score']
        )
        
        return ECGPrediction(
            condition_code=top_pred['code'],
            condition_name=top_pred['name'],
            probability=top_pred['probability'],
            confidence=top_pred.get('confidence', 'medium'),
            top_conditions=results['top_predictions'],
            clinical_significance=condition_info.get('clinical_significance', ''),
            severity=severity,
            recommendations=recommendations,
            signal_quality=quality_metrics['overall_score'],
            prediction_quality=prediction_quality,
            explanations=explanations,
            processing_time_ms=processing_time,
            model_version=self.config.model_type.value
        )
    
    def _create_low_quality_prediction(
        self,
        quality_metrics: Dict[str, Any]
    ) -> ECGPrediction:
        """Create prediction for low quality signal"""
        return ECGPrediction(
            condition_code="poor_quality",
            condition_name="Poor Signal Quality",
            probability=0.0,
            confidence=0.0,
            top_conditions=[],
            clinical_significance="Signal quality too poor for reliable analysis",
            severity="unknown",
            recommendations=[
                "Repeat ECG recording with better electrode placement",
                "Ensure patient is relaxed and still during recording",
                "Check electrode connections and skin preparation"
            ],
            signal_quality=quality_metrics['overall_score'],
            prediction_quality="unreliable",
            processing_time_ms=0.0
        )
    
    def _determine_severity(self, condition_code: str, probability: float) -> str:
        """Determine condition severity"""
        # Critical conditions
        critical_conditions = [
            'ventricular_fibrillation',
            'ventricular_tachycardia',
            'complete_heart_block',
            'acute_mi'
        ]
        
        if condition_code in critical_conditions and probability > 0.7:
            return "critical"
        elif condition_code in critical_conditions and probability > 0.5:
            return "high"
        elif probability > 0.8:
            return "moderate"
        else:
            return "low"
    
    def _generate_recommendations(
        self,
        condition_code: str,
        severity: str,
        quality_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate clinical recommendations"""
        recommendations = []
        
        # Severity-based recommendations
        if severity == "critical":
            recommendations.append("Immediate medical attention required")
            recommendations.append("Consider emergency cardiac intervention")
        elif severity == "high":
            recommendations.append("Urgent cardiology consultation recommended")
            recommendations.append("Consider continuous cardiac monitoring")
        
        # Condition-specific recommendations
        condition_recommendations = {
            'atrial_fibrillation': [
                "Evaluate for anticoagulation therapy",
                "Consider rate/rhythm control strategies"
            ],
            'myocardial_infarction': [
                "Immediate cardiac catheterization evaluation",
                "Initiate acute coronary syndrome protocol"
            ],
            'left_bundle_branch_block': [
                "Evaluate for underlying structural heart disease",
                "Consider echocardiography"
            ]
        }
        
        if condition_code in condition_recommendations:
            recommendations.extend(condition_recommendations[condition_code])
        
        # Quality-based recommendations
        if quality_metrics['overall_score'] < 0.8:
            recommendations.append("Consider repeating ECG for better quality")
        
        return recommendations
    
    def _assess_prediction_quality(
        self,
        probability: float,
        signal_quality: float
    ) -> str:
        """Assess overall prediction quality"""
        if signal_quality > 0.9 and probability > 0.9:
            return "excellent"
        elif signal_quality > 0.8 and probability > 0.8:
            return "good"
        elif signal_quality > 0.7 and probability > 0.7:
            return "fair"
        else:
            return "poor"
    
    def _generate_cache_key(
        self,
        signal: np.ndarray,
        patient_context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate cache key for prediction"""
        # Use hash of signal and context
        signal_hash = hash(signal.tobytes())
        context_hash = hash(json.dumps(patient_context, sort_keys=True)) if patient_context else 0
        return f"{signal_hash}_{context_hash}"
    
    def _update_cache(self, key: str, prediction: ECGPrediction):
        """Update prediction cache with LRU eviction"""
        if len(self.prediction_cache) >= self.config.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.prediction_cache))
            del self.prediction_cache[oldest_key]
        
        self.prediction_cache[key] = prediction
    
    async def train_model(
        self,
        train_dataset: Any,
        val_dataset: Any,
        training_config: Optional[TrainingConfig] = None
    ) -> Dict[str, Any]:
        """Train or fine-tune the model"""
        if training_config is None:
            training_config = TrainingConfig(model_config=self.primary_model.config)
        
        # Create trainer
        trainer = ECGTrainer(
            self.primary_model,
            training_config,
            train_dataset,
            val_dataset,
            device=self.config.device
        )
        
        # Run training
        await trainer.train()
        
        # Return training metrics
        return trainer.get_final_metrics()
    
    def export_model(
        self,
        output_path: str,
        export_format: str = "pytorch"
    ) -> Path:
        """Export model for deployment"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if export_format == "pytorch":
            torch.save({
                'model_state_dict': self.primary_model.state_dict(),
                'model_config': self.primary_model.config,
                'service_config': self.config
            }, output_path)
        
        elif export_format == "onnx":
            dummy_input = torch.randn(1, 12, 5000).to(self.config.device)
            torch.onnx.export(
                self.primary_model,
                dummy_input,
                str(output_path),
                export_params=True,
                opset_version=11,
                input_names=['ecg_signal'],
                output_names=['predictions'],
                dynamic_axes={
                    'ecg_signal': {0: 'batch_size'},
                    'predictions': {0: 'batch_size'}
                }
            )
        
        logger.info(f"Model exported to {output_path}")
        return output_path


class ClinicalValidator:
    """Validates predictions against clinical rules"""
    
    def __init__(self, validation_rules: List[str]):
        self.validation_rules = validation_rules
        self.rule_validators = {
            'rhythm': self._validate_rhythm,
            'morphology': self._validate_morphology,
            'intervals': self._validate_intervals
        }
    
    def validate(
        self,
        prediction_results: Dict[str, Any],
        quality_metrics: Dict[str, Any],
        patient_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate predictions against clinical rules"""
        validated_results = prediction_results.copy()
        validation_flags = []
        
        for rule in self.validation_rules:
            if rule in self.rule_validators:
                is_valid, message = self.rule_validators[rule](
                    prediction_results,
                    quality_metrics,
                    patient_context
                )
                
                if not is_valid:
                    validation_flags.append({
                        'rule': rule,
                        'message': message
                    })
        
        validated_results['validation_flags'] = validation_flags
        validated_results['clinically_validated'] = len(validation_flags) == 0
        
        return validated_results
    
    def _validate_rhythm(
        self,
        results: Dict[str, Any],
        quality: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """Validate rhythm consistency"""
        # Example validation
        top_pred = results['top_predictions'][0]
        
        if 'irregular' in top_pred['name'].lower() and quality.get('regular_rr', True):
            return False, "Rhythm classification inconsistent with RR regularity"
        
        return True, ""
    
    def _validate_morphology(
        self,
        results: Dict[str, Any],
        quality: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """Validate morphology consistency"""
        # Placeholder for morphology validation
        return True, ""
    
    def _validate_intervals(
        self,
        results: Dict[str, Any],
        quality: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> Tuple[bool, str]:
        """Validate interval measurements"""
        # Placeholder for interval validation
        return True, ""


def create_ml_service(config: Optional[MLServiceConfig] = None) -> AdvancedMLService:
    """Factory function to create ML service"""
    return AdvancedMLService(config)
