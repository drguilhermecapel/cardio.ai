#!/usr/bin/env python
"""Script to run tests with coverage and ensure 80%+ coverage."""

import subprocess
import sys
import os


def main():
    """Run tests with coverage."""
    print("🧪 Running CardioAI Pro Backend Tests with Coverage...")
    print("=" * 60)

    # Set environment variables
    os.environ["ENVIRONMENT"] = "test"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    # Run poetry lock if needed
    print("📦 Updating dependencies...")
    subprocess.run(["poetry", "lock", "--no-update"], check=False)
    subprocess.run(["poetry", "install"], check=True)

    # Run tests with coverage
    print("\n🏃 Running tests...")
    cmd = [
        "poetry",
        "run",
        "pytest",
        "--cov=app",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "--cov-fail-under=80",
        "-v",
        "--tb=short",
        "tests/test_comprehensive_coverage_final.py",
    ]

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n✅ SUCCESS: Coverage requirement met (80%+)")
        print("📊 HTML report available at: htmlcov/index.html")
        print("\nTo view the report:")
        print("  - Windows: start htmlcov/index.html")
        print("  - Mac/Linux: open htmlcov/index.html")
    else:
        print("\n❌ FAILED: Coverage requirement not met or tests failed")
        print("Run 'poetry run pytest --cov=app --cov-report=html -v' for details")
        sys.exit(1)


if __name__ == "__main__":
    main()
