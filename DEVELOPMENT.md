# Development Guide

This guide covers setting up a local development environment for contributing to `smolagents`.

---

## Prerequisites

- Python 3.10 or newer
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- `git`

---

## Installation

### Using `uv` (recommended)

```bash
git clone https://github.com/huggingface/smolagents.git
cd smolagents
uv pip install -e ".[dev]"
```

### Using `pip`

```bash
git clone https://github.com/huggingface/smolagents.git
cd smolagents
pip install -e ".[dev]"
```

The `-e` flag installs the package in *editable* mode so changes to source files are immediately reflected without reinstalling.

---

## Running Tests

The test suite lives in `tests/`. Run it with:

```bash
# Using Make (recommended)
make test

# Direct pytest
pytest tests/

# Run a single test file
pytest tests/test_agents.py -v

# Run a single test by name
pytest tests/test_agents.py -k "test_code_agent" -v
```

### Test markers

| Marker | Purpose |
|--------|---------|
| `slow` | Tests that hit real LLM APIs — skipped in CI by default |
| `e2b`  | Tests requiring an E2B sandbox API key |

Skip slow tests locally:
```bash
pytest tests/ -m "not slow"
```

---

## Linting and Formatting

We use [`ruff`](https://docs.astral.sh/ruff/) for both linting and formatting, and [`pre-commit`](https://pre-commit.com/) to run checks automatically on commit.

### Install pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

### Run all checks manually

```bash
pre-commit run --all-files
```

### Run individual tools

```bash
# Format code
ruff format src/ tests/

# Check for lint errors
ruff check src/ tests/

# Auto-fix lint errors where possible
ruff check --fix src/ tests/
```

---

## Project Structure

```text
smolagents/
├── src/
│   └── smolagents/          # Main library source
│       ├── agents.py         # Agent classes (CodeAgent, ToolCallingAgent, …)
│       ├── tools.py          # Base Tool class and built-in tools
│       ├── models.py         # LLM model wrappers
│       ├── memory.py         # Agent memory / message history
│       ├── monitoring.py     # Logging and step callbacks
│       └── utils.py          # Shared utilities
├── tests/                   # Pytest test suite
├── examples/                # Standalone usage examples
├── docs/                    # Documentation source (MkDocs)
├── pyproject.toml           # Package metadata and dependencies
└── Makefile                 # Common development targets
```

---

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make test` | Run the full test suite |
| `make lint` | Run ruff linter |
| `make format` | Apply ruff formatter |
| `make docs` | Build the documentation locally |

---

## Writing a New Tool

1. Subclass `Tool` from `smolagents.tools`.
2. Declare `name`, `description`, and `inputs` / `output_type` as class attributes.
3. Implement the `forward(self, ...)` method with a return type annotation.
4. Add a test in `tests/` that exercises the happy path and at least one error case.

```python
from smolagents import Tool

class ReverseStringTool(Tool):
    name = "reverse_string"
    description = "Reverses the characters in a string."
    inputs = {"text": {"type": "string", "description": "The string to reverse."}}
    output_type = "string"

    def forward(self, text: str) -> str:
        return text[::-1]
```

---

## Submitting a Pull Request

1. Fork the repo and create a feature branch from `main`.
2. Make your changes with tests.
3. Run `pre-commit run --all-files` and fix any issues.
4. Open a PR targeting `main` with a clear description of *what changed* and *why*.

For larger changes, open a GitHub Issue first to discuss the design before writing code.
