# smolagents Development Guide

## Setup

```bash
git clone https://github.com/VANDRANKI/smolagents.git
cd smolagents
pip install -e ".[dev]"
# or
uv sync
```

## Running Tests

```bash
# Full suite
make test

# Or directly
pytest tests/ -v

# Single file
pytest tests/test_agents.py -v

# Match by name
pytest tests/ -k "test_tool_call" -v
```

## Code Quality

```bash
# Format (ruff)
uv run ruff format .

# Lint
uv run ruff check .

# Pre-commit (recommended)
pre-commit install
pre-commit run --all-files
```

## Type Hints

All public functions must have complete type annotations:

```python
from typing import Any, Optional

class Tool:
    def run(
        self,
        *args: Any,
        sanitize_inputs_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Execute the tool with the given arguments."""
        ...
```

## Docstrings

Use Google-style docstrings. Do not repeat type information that is
already present in the function signature.

```python
def load_tool(task: str, model_id: Optional[str] = None) -> Tool:
    """Load a community tool for the given task.

    Args:
        task: A short description of the tool's purpose (e.g. 'translation').
        model_id: Optional HuggingFace model ID to override the default.

    Returns:
        An instantiated Tool ready for use in an agent.

    Raises:
        ValueError: If no tool is found for the given task.
    """
```

## Adding a New Tool

1. Subclass `Tool` in `src/smolagents/tools.py` (or a new file)
2. Define `name`, `description`, and `inputs` class attributes
3. Implement `forward()` with full type hints
4. Add a unit test in `tests/test_tools.py`
5. Document it in `docs/`

## Commit Convention

```
feat: add WebSearchTool with DuckDuckGo backend
fix: handle empty output in CodeAgent.run
docs: clarify streaming usage in README
test: add coverage for ManagedAgent.handoff
```
