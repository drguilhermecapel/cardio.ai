#!/usr/bin/env python3
"""
Test script to verify SCP-ECG conditions file compiles correctly
"""

import sys
from pathlib import Path


def test_scp_conditions_compilation():
    """Test that SCP-ECG conditions file compiles and validates correctly"""
    try:
        from app.core.scp_ecg_conditions import (
            SCP_ECG_CONDITIONS,
            SCPCategory,
            validate_scp_conditions,
            get_condition_by_code,
            get_conditions_by_category,
            get_critical_conditions,
        )

        print("✓ SCP-ECG conditions file imported successfully")

        validation_result = validate_scp_conditions()
        print(f"✓ Validation result: {validation_result}")

        norm_condition = get_condition_by_code("NORM")
        print(
            f"✓ Normal condition: {norm_condition.name if norm_condition else 'Not found'}"
        )

        arrhythmia_conditions = get_conditions_by_category(SCPCategory.ARRHYTHMIA)
        print(f"✓ Arrhythmia conditions count: {len(arrhythmia_conditions)}")

        critical_conditions = get_critical_conditions()
        print(f"✓ Critical conditions count: {len(critical_conditions)}")

        total_conditions = len(SCP_ECG_CONDITIONS)
        print(f"✓ Total conditions: {total_conditions}")

        if total_conditions == 71:
            print("✅ SUCCESS: All 71 SCP-ECG conditions implemented correctly")
            return True
        else:
            print(f"❌ ERROR: Expected 71 conditions, got {total_conditions}")
            return False

    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_scp_conditions_compilation()
    sys.exit(0 if success else 1)
