# Custom Tool Creation Guide

This guide covers how to create well-structured custom tools for smolagents.

## Basic Tool Structure

A tool is a Python function decorated with `@tool`. The docstring and type
annotations are parsed to build the tool schema shown to the model.

```python
from smolagents import tool


@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather for a city.

    Args:
        city: The city name to get weather for.
        unit: Temperature unit, either 'celsius' or 'fahrenheit'.

    Returns:
        A string describing the current weather conditions.
    """
    # Validate inputs at the boundary
    unit = unit.lower()
    if unit not in ("celsius", "fahrenheit"):
        raise ValueError(f"Invalid unit '{unit}'. Must be 'celsius' or 'fahrenheit'.")
    if not city.strip():
        raise ValueError("City name cannot be empty.")

    # ... actual implementation ...
    return f"Weather in {city}: 22°{'C' if unit == 'celsius' else 'F'}, sunny"
```

## Type Annotations Matter

Annotations directly determine the JSON schema the model receives:

| Python Type | JSON Schema |
|---|---|
| `str` | `{"type": "string"}` |
| `int` | `{"type": "integer"}` |
| `float` | `{"type": "number"}` |
| `bool` | `{"type": "boolean"}` |
| `list[str]` | `{"type": "array", "items": {"type": "string"}}` |

**Always annotate both parameters and return type.** Missing annotations produce
vague schemas that confuse the model.

## Error Handling Best Practices

```python
@tool
def search_database(query: str, limit: int = 10) -> list[dict]:
    """Search the product database.

    Args:
        query: Search query string.
        limit: Maximum number of results to return (1-100).

    Returns:
        List of matching product dictionaries.
    """
    if not query.strip():
        raise ValueError("Search query cannot be empty.")
    if not 1 <= limit <= 100:
        raise ValueError(f"limit must be between 1 and 100, got {limit}.")

    try:
        results = db.search(query, limit=limit)
    except ConnectionError as e:
        # Re-raise with context so the agent can inform the user
        raise RuntimeError(f"Database unavailable: {e}") from e

    return [{"id": r.id, "name": r.name, "price": r.price} for r in results]
```

## Testing Tools in Isolation

Always test your tool before connecting it to an agent:

```python
import pytest
from mytools import search_database


def test_search_database_returns_results():
    results = search_database("laptop", limit=5)
    assert len(results) <= 5
    assert all("id" in r and "name" in r for r in results)


def test_search_database_rejects_empty_query():
    with pytest.raises(ValueError, match="cannot be empty"):
        search_database("")


def test_search_database_rejects_invalid_limit():
    with pytest.raises(ValueError, match="between 1 and 100"):
        search_database("laptop", limit=0)
```

## Common Pitfalls

| Pitfall | Problem | Fix |
|---|---|---|
| Missing return annotation | Vague schema | Always annotate return type |
| Catching all exceptions | Hides bugs | Only catch specific exceptions |
| Silent failures | Agent gets `None` and loops | Raise descriptive errors |
| No input validation | Model can pass bad values | Validate at the top of each tool |
| Side effects in tests | Flaky tests | Mock external calls |
