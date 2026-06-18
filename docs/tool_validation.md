# smolagents Tool Validation Guide

This guide covers how to write robust `Tool` subclasses and how to catch
common errors — especially `forward()` signature mismatches — before
they fail at runtime.

## Tool Anatomy

```python
from smolagents import Tool

class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web for current information on a topic."
    inputs = {
        "query": {"type": "string", "description": "The search query."},
        "max_results": {
            "type": "integer",
            "description": "Maximum results to return.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, query: str, max_results: int = 5) -> str:
        """Execute the web search and return formatted results.

        Args:
            query: The search query string.
            max_results: Maximum number of results to include.

        Returns:
            A formatted string of search results.
        """
        results = self._search_api(query, n=max_results)
        return "\n".join(f"- {r.title}: {r.url}" for r in results)
```

## The Signature Mismatch Bug

The most common bug in `Tool` subclasses is when the `inputs` schema
declares different parameter names than `forward()` accepts:

```python
# WRONG: inputs says 'search_query' but forward() expects 'query'
class BrokenTool(Tool):
    inputs = {"search_query": {"type": "string", "description": "..."}}

    def forward(self, query: str) -> str:  # mismatch! will fail at runtime
        ...
```

This fails with `TypeError: forward() got an unexpected keyword argument 'search_query'`
only when the agent actually calls the tool — not during import or initialization.

## Validating Tools in Tests

Add a test that calls your tool's `forward()` directly to catch signature
mismatches before they reach the agent:

```python
import inspect
import pytest
from mypackage.tools import WebSearchTool

def test_web_search_tool_signature_matches_inputs():
    tool = WebSearchTool()
    sig = inspect.signature(tool.forward)
    params = set(sig.parameters.keys())

    # Every required input should appear in forward()
    for input_name, input_spec in tool.inputs.items():
        if not input_spec.get("nullable", False):
            assert input_name in params, (
                f"forward() is missing required input parameter '{input_name}'. "
                f"Add it to forward() or mark it nullable in inputs."
            )

def test_web_search_tool_returns_string():
    tool = WebSearchTool()
    result = tool.forward(query="Python type hints", max_results=2)
    assert isinstance(result, str)
    assert len(result) > 0
```

## Type Annotations on `forward()`

Always annotate `forward()` with full type hints:

```python
from typing import Optional

def forward(self, path: str, encoding: Optional[str] = None) -> str:
    ...
```

The return type must match `output_type`:

| `output_type` value | Expected Python return type |
|---------------------|-----------------------------|
| `"string"`          | `str`                       |
| `"integer"`         | `int`                       |
| `"number"`          | `float`                     |
| `"boolean"`         | `bool`                      |
| `"object"`          | `dict`                      |
| `"image"`           | `PIL.Image.Image`           |
| `"audio"`           | `bytes` or file path `str`  |

## Nullable Inputs

For optional tool parameters, mark them `nullable: True` in `inputs` AND
give them a default in `forward()`:

```python
class SearchTool(Tool):
    inputs = {
        "query": {"type": "string", "description": "Search query."},
        "language": {
            "type": "string",
            "description": "Result language code, e.g. 'en'.",
            "nullable": True,     # agent may omit this
        },
    }

    def forward(self, query: str, language: Optional[str] = None) -> str:
        lang = language or "en"
        ...
```

## Testing Without an LLM

Test tools in isolation — never spin up a full agent just to test a tool:

```python
def test_calculator_tool():
    tool = CalculatorTool()
    assert tool.forward(expression="2 + 2") == "4"
    assert tool.forward(expression="10 / 0") == "Error: division by zero"
```
