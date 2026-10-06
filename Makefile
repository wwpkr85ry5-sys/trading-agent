# Local development commands

.PHONY: install install-dev test lint fmt clean help

help:
	@echo "Available commands:"
	@echo "  make install        - Install the package"
	@echo "  make install-dev    - Install the package with dev dependencies"
	@echo "  make test           - Run tests"
	@echo "  make lint           - Run linting checks"
	@echo "  make clean          - Clean up build artifacts"

install:
	python -m pip install -e .

install-dev:
	python -m pip install -e . -r requirements.txt

test:
	pytest -v

lint:
	python -m flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .cache -exec rm -rf {} +
