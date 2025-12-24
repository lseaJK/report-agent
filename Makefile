# Investment Research Reports System Makefile

.PHONY: help install dev-install test lint format type-check clean run migrate

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install production dependencies"
	@echo "  dev-install  Install development dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  type-check   Run type checking"
	@echo "  clean        Clean up temporary files"
	@echo "  run          Run the application"
	@echo "  migrate      Run database migrations"

# Install production dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
dev-install:
	pip install -r requirements.txt
	pip install -e ".[dev]"

# Run tests
test:
	pytest tests/ -v

# Run linting
lint:
	flake8 src/ tests/
	isort --check-only src/ tests/
	black --check src/ tests/

# Format code
format:
	isort src/ tests/
	black src/ tests/

# Run type checking
type-check:
	mypy src/

# Clean up temporary files
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf dist/
	rm -rf build/

# Run the application
run:
	python -m src.investment_research.main

# Run database migrations
migrate:
	alembic upgrade head

# Create new migration
migration:
	alembic revision --autogenerate -m "$(MSG)"

# Development setup
setup: dev-install migrate
	@echo "Development environment setup complete!"

# CI/CD pipeline
ci: lint type-check test
	@echo "CI pipeline completed successfully!"