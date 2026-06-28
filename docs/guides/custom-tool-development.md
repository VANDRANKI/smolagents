# Custom Tool Development Guide

This guide explains how to write custom tools for smolagents, covering the
decorator API, the class-based API, input validation, and error handling.

## The `@tool` Decorator

The quickest way to expose a Python function as an agent tool:

```python
from smolagents import tool


@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    """Return the current weather for a given city.

    Args:
        city: The name of the city to look up (e.g. 'London', 'Tokyo').
        unit: Temperature unit — 'celsius' or 'fahrenheit'. Defaults to 'celsius'.

    Returns:
        A plain-text weather summary string.
    """
    # Real implementation would call a weather API.
    return f"The weather in {city} is 22 {unit}."
```

**Requirements for the decorator:**
- The function **must** have a Google-style docstring — smolagents uses it to
  build the tool description the LLM sees.
- All parameters **must** have type annotations. Supported types: `str`, `int`,
  `float`, `bool`, `list`, `dict`.
- The return type **must** be annotated.

## Class-Based Tool

For tools that need shared state (e.g. a database connection, a cached HTTP
session), subclass `Tool`:

```python
from typing import Any
from smolagents import Tool


class DatabaseQueryTool(Tool):
    """Execute read-only SQL queries against the application database."""

    name = "database_query"
    description = (
        "Run a read-only SQL SELECT query and return the results as a "
        "JSON-serialisable list of dicts. Only SELECT statements are allowed."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "A SQL SELECT statement to execute.",
        }
    }
    output_type = "string"

    def __init__(self, connection_string: str) -> None:
        super().__init__()
        import sqlite3
        self._conn = sqlite3.connect(connection_string)
        self._conn.row_factory = sqlite3.Row

    def forward(self, query: str) -> str:
        """Execute the query and return JSON-formatted results.

        Args:
            query: A read-only SQL SELECT statement.

        Returns:
            JSON string containing the query results as a list of objects.

        Raises:
            ValueError: If the query is not a SELECT statement.
        """
        import json

        stripped = query.strip().upper()
        if not stripped.startswith("SELECT"):
            raise ValueError(
                f"Only SELECT queries are permitted. Got: {query[:50]!r}"
            )
        cursor = self._conn.execute(query)
        rows = [dict(row) for row in cursor.fetchall()]
        return json.dumps(rows, default=str)
```

## Input Validation Pattern

Validate inputs at the top of `forward` before touching external systems:

```python
def forward(self, url: str, timeout: int = 10) -> str:
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"url must start with http:// or https://. Got: {url!r}")
    if not 1 <= timeout <= 120:
        raise ValueError(f"timeout must be between 1 and 120. Got: {timeout}")
    # ... rest of implementation
```

## Testing Your Tool in Isolation

Before wiring a tool into an agent, test it directly:

```python
tool = DatabaseQueryTool("sqlite:///test.db")

# Happy path
result = tool.forward("SELECT id, name FROM users LIMIT 3")
print(result)  # JSON string

# Error path
try:
    tool.forward("DROP TABLE users")
except ValueError as e:
    print(f"Caught expected error: {e}")
```

## Registering Tools with an Agent

```python
from smolagents import CodeAgent, HfApiModel

model = HfApiModel(model_id="Qwen/Qwen2.5-72B-Instruct")

agent = CodeAgent(
    tools=[
        get_weather,
        DatabaseQueryTool("app.db"),
    ],
    model=model,
    max_steps=5,
)

response = agent.run("What is the weather in Paris right now?")
```

## Common Mistakes

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Missing docstring | `ValueError: Tool must have a docstring` | Add Google-style docstring with Args section |
| Unannotated parameters | Tool schema generation fails | Add type annotations to every parameter |
| `forward` raises unexpected types | Agent retries indefinitely | Raise `ValueError` or `RuntimeError` with a clear message |
| Returning `None` | LLM sees `null` output | Return an empty string `""` or a status string instead |
| Mutable default argument in `inputs` | State shared across calls | Use `None` default and set inside `__init__` |
