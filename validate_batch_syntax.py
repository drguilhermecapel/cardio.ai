#!/usr/bin/env python3
"""
Validate Windows batch script syntax for potential parsing issues
"""

import re
import sys
from pathlib import Path


def validate_batch_syntax(filepath):
    """Check for common Windows batch syntax issues"""
    print(f"Validating {filepath}...")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    issues = []
    lines = content.split("\n")

    for i, line in enumerate(lines, 1):
        line = line.strip()

        if (
            "node" in line.lower()
            and "-e" in line
            and '"' not in line
            and "powershell" not in line.lower()
        ):
            issues.append(f"Line {i}: Unquoted Node.js command: {line}")

        if (
            any(keyword in line for keyword in ["from", "import", "slice"])
            and "echo" not in line.lower()
            and 'python -c "' not in line
            and "powershell" not in line.lower()
        ):
            issues.append(f"Line {i}: Potential keyword conflict: {line}")

    if issues:
        print("❌ Potential syntax issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ No obvious syntax issues found")
        return True


if __name__ == "__main__":
    script_path = Path(__file__).parent / "test_node_version_check.bat"
    success = validate_batch_syntax(str(script_path))
    sys.exit(0 if success else 1)
