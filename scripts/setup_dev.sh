#!/usr/bin/env bash
# setup_dev.sh — bootstrap a local development environment for smolagents
#
# Usage:
#   bash scripts/setup_dev.sh
#
# What this script does:
#   1. Creates a virtual environment at .venv/ (if it doesn't already exist)
#   2. Installs the package in editable mode with dev dependencies
#   3. Installs pre-commit hooks so linting runs automatically on every commit

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

echo "==> Working directory: ${ROOT_DIR}"
cd "${ROOT_DIR}"

# ---------------------------------------------------------------------------
# 1. Create virtual environment
# ---------------------------------------------------------------------------
if [ ! -d "${VENV_DIR}" ]; then
    echo "==> Creating virtual environment at .venv/"
    python3 -m venv "${VENV_DIR}"
else
    echo "==> Virtual environment already exists at .venv/ — skipping creation"
fi

# Activate the venv for the remainder of the script
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ---------------------------------------------------------------------------
# 2. Install the package in editable mode with dev dependencies
# ---------------------------------------------------------------------------
echo "==> Installing smolagents in editable mode with dev dependencies"

# Prefer uv for speed; fall back to pip
if command -v uv &>/dev/null; then
    echo "    (using uv)"
    uv pip install -e ".[dev]"
else
    echo "    (uv not found, using pip — consider installing uv for faster installs)"
    pip install --upgrade pip
    pip install -e ".[dev]"
fi

# ---------------------------------------------------------------------------
# 3. Install pre-commit hooks
# ---------------------------------------------------------------------------
if command -v pre-commit &>/dev/null; then
    echo "==> Installing pre-commit hooks"
    pre-commit install
else
    echo "==> pre-commit not found — installing it now"
    pip install pre-commit
    pre-commit install
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "Development environment is ready!"
echo ""
echo "Next steps:"
echo "  source .venv/bin/activate   # activate the virtual environment"
echo "  make test                   # run the test suite"
echo "  pre-commit run --all-files  # run all linting checks"
