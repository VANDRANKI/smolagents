# Contributing to smolagents

Thank you for contributing to smolagents! This guide covers setup, code standards, and testing.

## Setup

```bash
pip install -e ".[dev]"
pre-commit install
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run a specific test
pytest tests/test_agents.py -v

# Run with coverage
pytest tests/ --cov=smolagents --cov-report=term-missing
```

## Code Standards

### Error Handling

Use specific exceptions rather than broad `Exception` catches:

```python
# Good: specific exception handling
try:
    result = agent.run(task)
except TimeoutError as e:
    logger.error("Agent timed out", timeout=e.timeout, task=task)
    raise
except ValueError as e:
    logger.error("Invalid task format", error=str(e))
    raise AgentError(f"Failed to run task: {e}") from e

# Bad: bare except
try:
    result = agent.run(task)
except:
    pass  # Never do this
```

### Type Hints

All public functions must have complete type annotations:

```python
from typing import Optional, Union

def create_agent(
    model: str,
    tools: Optional[list] = None,
    max_steps: int = 10,
    verbose: bool = False,
) -> "CodeAgent":
    """Create a new CodeAgent instance.

    Args:
        model: Model ID string (e.g. 'gpt-4o').
        tools: Optional list of tool instances.
        max_steps: Maximum number of reasoning steps.
        verbose: Whether to print step-by-step output.

    Returns:
        A configured CodeAgent ready to run tasks.
    """
    ...
```

## Agent Execution Patterns

When implementing custom agents, always:

1. Validate inputs before execution starts
2. Log each reasoning step for debuggability
3. Handle tool errors gracefully (don't crash the agent)
4. Respect `max_steps` to prevent infinite loops
5. Return structured output when possible

## Submitting Changes

1. Fork the repo and create a feature branch
2. Make your changes with tests
3. Run `pre-commit run --all-files`
4. Open a PR with a clear description of what changed and why
