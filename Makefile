.PHONY: install install-dev test test-unit test-integration test-quick clean download-models run-mamba1 run-mamba2 format show-structure help

# Default Python command
PYTHON := uv run python

# Installation
install:
	uv sync
	uv pip install -e .

# Testing
test: test-unit test-integration

test-unit:
	$(PYTHON) -m tests.run_all_tests

test-integration:
	$(PYTHON) -m tests.integration.test_unified_system

test-quick:
	$(PYTHON) -m tests.integration.test_unified_system

# Model operations
download-models:
	$(PYTHON) -m scripts.download_models mamba1
	$(PYTHON) -m scripts.download_models mamba2

run-mamba1:
	$(PYTHON) -m scripts.run_models mamba1 --prompt "The future of AI"

run-mamba2:
	$(PYTHON) -m scripts.run_models mamba2 --prompt "The future of AI"

install-dev:
	uv sync --extra dev

# Development
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

format:
	uv run black src/ scripts/ tests/
	uv run isort src/ scripts/ tests/

# Documentation
show-structure:
	@echo "📚 Project structure:"
	@echo "src/mamba_macos/     - Core library code"
	@echo "scripts/             - Utility scripts"
	@echo "tests/unit/          - Unit tests"
	@echo "tests/integration/   - Integration tests"
	@echo "examples/            - Usage examples"
	@echo "config/              - Configuration files"
	@echo "models/              - Downloaded models"

help:
	@echo "Available commands:"
	@echo "  install           - Install dependencies and package"
	@echo "  test              - Run all tests"
	@echo "  test-unit         - Run unit tests only"
	@echo "  test-integration  - Run integration tests only"
	@echo "  test-quick        - Run quick test"
	@echo "  download-models   - Download both Mamba models"
	@echo "  run-mamba1        - Run Mamba1 model demo"
	@echo "  run-mamba2        - Run Mamba2 model demo"
	@echo "  clean             - Clean up cache files"
	@echo "  format            - Format code with black and isort"
	@echo "  show-structure    - Show project structure" 