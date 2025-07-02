#!/usr/bin/env python3
"""
Integration test for advanced preprocessing pipeline with existing system
"""

import sys
import os

sys.path.append(".")

import numpy as np
import time
from app.services.hybrid_ecg_service import HybridECGAnalysisService


def test_integration_with_existing_system():
    """Test that advanced preprocessing integrates correctly with HybridECGAnalysisService"""
    print("=" * 60)
    print("TESTING INTEGRATION WITH EXISTING SYSTEM")
    print("=" * 60)

    try:
        print("Initializing HybridECGAnalysisService...")
        service = HybridECGAnalysisService()
        print("✓ HybridECGAnalysisService initialized successfully")

        print("\nCreating test ECG data...")
        fs = 360
        duration = 10  # seconds
        t = np.linspace(0, duration, fs * duration)

        heart_rate = 75  # bpm
        beat_interval = 60 / heart_rate

        lead1 = np.zeros_like(t)
        for beat_time in np.arange(0, duration, beat_interval):
            beat_idx = int(beat_time * fs)
            if beat_idx < len(lead1) - 50:
                qrs_width = int(0.08 * fs)
                qrs_indices = np.arange(
                    beat_idx - qrs_width // 2, beat_idx + qrs_width // 2
                )
                qrs_indices = qrs_indices[
                    (qrs_indices >= 0) & (qrs_indices < len(lead1))
                ]
                qrs_pattern = np.exp(
                    -0.5 * ((qrs_indices - beat_idx) / (qrs_width / 6)) ** 2
                )
                lead1[qrs_indices] += qrs_pattern

        baseline = 0.05 * np.sin(2 * np.pi * 0.1 * t)
        noise = 0.02 * np.random.normal(0, 1, len(t))
        lead1 += baseline + noise

        signal_data = np.column_stack([lead1, lead1 * 0.8, lead1 * 0.6])  # 3 leads

        test_ecg_data = {
            "signal": signal_data,
            "sampling_rate": fs,
            "labels": ["Lead I", "Lead II", "Lead III"],
        }

        print(f"✓ Test ECG data created")
        print(f"  - Signal shape: {signal_data.shape}")
        print(f"  - Sampling rate: {fs} Hz")
        print(f"  - Duration: {duration} seconds")
        print(f"  - Number of leads: {len(test_ecg_data['labels'])}")

        print("\n" + "-" * 40)
        print("TESTING COMPREHENSIVE ECG ANALYSIS")
        print("-" * 40)

        start_time = time.time()

        result = service.analyze_ecg_comprehensive(
            ecg_data=test_ecg_data, patient_id="test_patient_001"
        )

        analysis_time = (time.time() - start_time) * 1000

        print("✓ Comprehensive ECG analysis completed successfully")
        print(f"  - Analysis time: {analysis_time:.1f}ms")

        expected_keys = [
            "signal_quality",
            "preprocessing_info",
            "features",
            "ai_analysis",
            "pathology_detection",
            "clinical_assessment",
        ]

        missing_keys = []
        for key in expected_keys:
            if key not in result:
                missing_keys.append(key)

        if missing_keys:
            print(f"⚠ Missing result keys: {missing_keys}")
        else:
            print("✓ All expected result keys present")

        if "signal_quality" in result:
            quality_info = result["signal_quality"]
            print(
                f"  - Signal quality score: {quality_info.get('overall_score', 'N/A')}"
            )
            print(f"  - Quality assessment: {quality_info.get('assessment', 'N/A')}")

        if "preprocessing_info" in result:
            prep_info = result["preprocessing_info"]
            print(f"  - Preprocessing method: {prep_info.get('method', 'N/A')}")
            print(
                f"  - Processing time: {prep_info.get('processing_time_ms', 'N/A')}ms"
            )

        if "ai_analysis" in result:
            ai_info = result["ai_analysis"]
            print(f"  - AI confidence: {ai_info.get('confidence', 'N/A')}")
            print(f"  - Primary finding: {ai_info.get('primary_finding', 'N/A')}")

        print("\n" + "-" * 40)
        print("TESTING SIMPLIFIED ECG ANALYSIS")
        print("-" * 40)

        start_time = time.time()

        simple_result = service.analyze_ecg_simple(
            ecg_data=test_ecg_data, patient_id="test_patient_002"
        )

        simple_analysis_time = (time.time() - start_time) * 1000

        print("✓ Simplified ECG analysis completed successfully")
        print(f"  - Analysis time: {simple_analysis_time:.1f}ms")
        print(f"  - Result keys: {list(simple_result.keys())}")

        print("\n" + "=" * 60)
        print("INTEGRATION TEST SUMMARY")
        print("=" * 60)

        success_criteria = []

        if result and len(result) > 0:
            print("✓ Comprehensive analysis integration successful")
            success_criteria.append(True)
        else:
            print("✗ Comprehensive analysis integration failed")
            success_criteria.append(False)

        if simple_result and len(simple_result) > 0:
            print("✓ Simplified analysis integration successful")
            success_criteria.append(True)
        else:
            print("✗ Simplified analysis integration failed")
            success_criteria.append(False)

        if analysis_time < 5000:  # Less than 5 seconds
            print(f"✓ Analysis performance acceptable: {analysis_time:.1f}ms")
            success_criteria.append(True)
        else:
            print(f"⚠ Analysis performance slow: {analysis_time:.1f}ms")
            success_criteria.append(False)

        success_rate = sum(success_criteria) / len(success_criteria) * 100
        print(f"\nOverall integration success rate: {success_rate:.1f}%")

        if success_rate >= 80:
            print("🎉 INTEGRATION TEST PASSED!")
            return True
        else:
            print("❌ INTEGRATION TEST FAILED!")
            return False

    except Exception as e:
        print(f"✗ Integration test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_integration_with_existing_system()
    sys.exit(0 if success else 1)
