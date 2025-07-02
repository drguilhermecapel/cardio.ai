"""
Test suite for advanced ECG preprocessing pipeline
"""

import os
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///test.db"

import numpy as np
import pytest
from app.preprocessing import AdvancedECGPreprocessor, EnhancedSignalQualityAnalyzer


class TestAdvancedECGPreprocessor:
    """Test cases for the advanced ECG preprocessing pipeline"""

    def setup_method(self):
        """Setup test fixtures"""
        self.preprocessor = AdvancedECGPreprocessor(sampling_rate=500)
        self.quality_analyzer = EnhancedSignalQualityAnalyzer(sampling_rate=500)

        self.fs = 500
        self.duration = 10  # seconds
        self.t = np.linspace(0, self.duration, self.fs * self.duration)

        self.clean_ecg = self._generate_synthetic_ecg()
        self.noisy_ecg = self._add_noise_to_ecg(self.clean_ecg)

    def _generate_synthetic_ecg(self):
        """Generate a synthetic ECG signal with realistic characteristics"""
        heart_rate = 70  # bpm
        beat_interval = 60 / heart_rate  # seconds

        ecg_signal = np.zeros_like(self.t)

        for beat_time in np.arange(0, self.duration, beat_interval):
            beat_idx = int(beat_time * self.fs)
            if beat_idx < len(ecg_signal) - 50:
                qrs_width = int(0.08 * self.fs)  # 80ms QRS width
                qrs_indices = np.arange(
                    beat_idx - qrs_width // 2, beat_idx + qrs_width // 2
                )
                qrs_indices = qrs_indices[
                    (qrs_indices >= 0) & (qrs_indices < len(ecg_signal))
                ]

                qrs_pattern = np.exp(
                    -0.5 * ((qrs_indices - beat_idx) / (qrs_width / 6)) ** 2
                )
                ecg_signal[qrs_indices] += qrs_pattern

        baseline = 0.1 * np.sin(2 * np.pi * 0.1 * self.t)  # 0.1 Hz baseline wander

        return ecg_signal + baseline

    def _add_noise_to_ecg(self, clean_signal):
        """Add various types of noise to clean ECG signal"""
        noisy_signal = clean_signal.copy()

        noise_level = 0.1
        white_noise = np.random.normal(0, noise_level, len(clean_signal))
        noisy_signal += white_noise

        powerline_noise = 0.05 * np.sin(2 * np.pi * 50 * self.t)
        noisy_signal += powerline_noise

        muscle_noise = 0.03 * np.random.normal(0, 1, len(clean_signal))
        muscle_noise = np.convolve(muscle_noise, np.ones(5) / 5, mode="same")  # Smooth
        noisy_signal += muscle_noise

        return noisy_signal

    def test_preprocessing_pipeline_basic(self):
        """Test basic functionality of preprocessing pipeline"""
        processed_signal, quality_metrics = (
            self.preprocessor.advanced_preprocessing_pipeline(
                self.clean_ecg, clinical_mode=True
            )
        )

        assert isinstance(processed_signal, np.ndarray)
        assert isinstance(quality_metrics, dict)
        assert len(processed_signal) > 0

        expected_keys = [
            "quality_score",
            "r_peaks_detected",
            "processing_time_ms",
            "segments_created",
            "meets_quality_threshold",
        ]
        for key in expected_keys:
            assert key in quality_metrics

        assert 0 <= quality_metrics["quality_score"] <= 1

        assert quality_metrics["r_peaks_detected"] > 0

    def test_preprocessing_with_noisy_signal(self):
        """Test preprocessing with noisy ECG signal"""
        processed_signal, quality_metrics = (
            self.preprocessor.advanced_preprocessing_pipeline(
                self.noisy_ecg, clinical_mode=True
            )
        )

        assert isinstance(processed_signal, np.ndarray)
        assert len(processed_signal) > 0

        assert quality_metrics["quality_score"] > 0

        assert quality_metrics["r_peaks_detected"] > 0

    def test_quality_assessment(self):
        """Test signal quality assessment"""
        clean_quality = self.preprocessor._assess_signal_quality_realtime(
            self.clean_ecg
        )
        assert 0 <= clean_quality <= 1

        noisy_quality = self.preprocessor._assess_signal_quality_realtime(
            self.noisy_ecg
        )
        assert 0 <= noisy_quality <= 1

        assert clean_quality >= noisy_quality

    def test_r_peak_detection(self):
        """Test R-peak detection accuracy"""
        r_peaks = self.preprocessor._pan_tompkins_detector(self.clean_ecg, self.fs)

        expected_peaks = int(70 * self.duration / 60)  # Approximately 12 peaks
        assert len(r_peaks) >= expected_peaks - 2  # Allow some tolerance
        assert len(r_peaks) <= expected_peaks + 2

        assert all(0 <= peak < len(self.clean_ecg) for peak in r_peaks)

    def test_filtering_methods(self):
        """Test individual filtering methods"""
        filtered = self.preprocessor._butterworth_bandpass_filter(
            self.noisy_ecg, 0.5, 40, self.fs
        )
        assert len(filtered) == len(self.noisy_ecg)
        assert not np.array_equal(filtered, self.noisy_ecg)  # Should be different

        denoised = self.preprocessor._wavelet_artifact_removal(self.noisy_ecg)
        assert len(denoised) == len(self.noisy_ecg)

    def test_segmentation_methods(self):
        """Test segmentation methods"""
        segments = self.preprocessor._fixed_window_segmentation(
            self.clean_ecg, window_seconds=5
        )
        assert len(segments) >= 1
        assert all(isinstance(seg, np.ndarray) for seg in segments)

        r_peaks = self.preprocessor._pan_tompkins_detector(self.clean_ecg, self.fs)
        if len(r_peaks) > 0:
            beat_segments = self.preprocessor._beat_by_beat_segmentation(
                self.clean_ecg, r_peaks, samples_per_beat=400
            )
            assert len(beat_segments) >= 1
            assert all(isinstance(seg, np.ndarray) for seg in beat_segments)

    def test_normalization(self):
        """Test robust normalization"""
        segments = [self.clean_ecg[:1000], self.clean_ecg[1000:2000]]
        normalized = self.preprocessor._robust_normalize(segments, method="median_iqr")

        assert isinstance(normalized, np.ndarray)
        assert len(normalized) > 0

        assert not np.array_equal(normalized, np.concatenate(segments))


class TestEnhancedSignalQualityAnalyzer:
    """Test cases for enhanced signal quality analyzer"""

    def setup_method(self):
        """Setup test fixtures"""
        self.analyzer = EnhancedSignalQualityAnalyzer(sampling_rate=500)

        self.fs = 500
        self.duration = 10
        self.t = np.linspace(0, self.duration, self.fs * self.duration)

        self.good_signal = np.sin(2 * np.pi * 1.2 * self.t)  # 1.2 Hz sine wave

        self.poor_signal = self.good_signal + 0.5 * np.random.normal(0, 1, len(self.t))

    def test_comprehensive_quality_assessment(self):
        """Test comprehensive quality assessment"""
        good_metrics = self.analyzer.assess_signal_quality_comprehensive(
            self.good_signal
        )

        expected_keys = [
            "overall_score",
            "lead_scores",
            "noise_characteristics",
            "artifact_detection",
            "frequency_analysis",
            "morphology_assessment",
            "recommendations",
        ]
        for key in expected_keys:
            assert key in good_metrics

        assert 0 <= good_metrics["overall_score"] <= 1

        assert isinstance(good_metrics["recommendations"], list)

        poor_metrics = self.analyzer.assess_signal_quality_comprehensive(
            self.poor_signal
        )

        assert poor_metrics["overall_score"] <= good_metrics["overall_score"]

    def test_noise_characterization(self):
        """Test noise characterization"""
        noise_char = self.analyzer._characterize_noise(self.poor_signal.reshape(-1, 1))

        expected_keys = [
            "powerline_interference",
            "baseline_wander",
            "muscle_artifact",
            "electrode_noise",
            "dominant_noise_type",
        ]
        for key in expected_keys:
            assert key in noise_char

        for key in expected_keys[:-1]:  # Exclude dominant_noise_type
            assert 0 <= noise_char[key] <= 1

    def test_artifact_detection(self):
        """Test artifact detection"""
        artifacts = self.analyzer._detect_artifacts(self.poor_signal.reshape(-1, 1))

        expected_keys = [
            "saturation_detected",
            "flat_line_segments",
            "sudden_jumps",
            "periodic_interference",
            "lead_disconnection",
        ]
        for key in expected_keys:
            assert key in artifacts

    def test_frequency_analysis(self):
        """Test frequency domain analysis"""
        freq_analysis = self.analyzer._analyze_frequency_domain(
            self.good_signal.reshape(-1, 1)
        )

        expected_keys = [
            "spectral_entropy",
            "dominant_frequency",
            "bandwidth_90",
            "ecg_band_power_ratio",
        ]
        for key in expected_keys:
            assert key in freq_analysis

        assert freq_analysis["dominant_frequency"] > 0
        assert 0 <= freq_analysis["ecg_band_power_ratio"] <= 1


if __name__ == "__main__":
    pytest.main([__file__])
