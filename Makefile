.PHONY: test test-coverage test-coverage-html install-test-deps

# Install test dependencies including pyedflib
install-test-deps:
	poetry add pyedflib
	poetry install --with dev

# Run all tests
test:
	poetry run pytest -v

# Run tests with coverage report
test-coverage:
	poetry run pytest --cov=app --cov-report=term-missing --cov-fail-under=80

# Run tests with HTML coverage report
test-coverage-html:
	poetry run pytest --cov=app --cov-report=html --cov-report=term-missing --cov-fail-under=80
	@echo "Coverage report generated at htmlcov/index.html"

# Run comprehensive coverage test
test-comprehensive:
	chmod +x run_coverage_test.sh
	./run_coverage_test.sh

# Clean test artifacts
clean-test:
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} +
