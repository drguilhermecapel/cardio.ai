#!/bin/bash

# Script to run tests with coverage and ensure 80%+ coverage

echo "Installing dependencies including pyedflib..."
poetry add pyedflib

echo "Running comprehensive test coverage..."

# Set environment variables
export ENVIRONMENT=test
export SKIP_DB_TESTS=false
export SKIP_API_TESTS=false
export SKIP_SLOW_TESTS=false

# Run tests with coverage
poetry run pytest \
    --cov=app \
    --cov-report=html \
    --cov-report=term-missing \
    --cov-report=xml \
    --cov-fail-under=80 \
    -v \
    tests/test_comprehensive_80_percent_coverage.py

# Check if coverage meets requirement
if [ $? -eq 0 ]; then
    echo "✅ Coverage requirement met (80%+)"
    echo "📊 HTML report available at: htmlcov/index.html"
else
    echo "❌ Coverage requirement not met"
    echo "Run 'poetry run pytest --cov=app --cov-report=html' to see detailed report"
    exit 1
fi
