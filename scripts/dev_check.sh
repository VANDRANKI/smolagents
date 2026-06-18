#!/usr/bin/env bash
# Run all quality checks locally before pushing.
# Usage: ./scripts/dev_check.sh
set -euo pipefail

echo "==> Formatting..."
uv run ruff format .

echo "==> Linting..."
uv run ruff check .

echo "==> Running tests..."
make test

echo ""
echo "All checks passed!"
