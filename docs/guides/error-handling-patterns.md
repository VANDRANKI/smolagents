# Error Handling Patterns in smolagents

This guide covers recommended patterns for handling errors in agent execution pipelines.

## Why Consistent Error Handling Matters

Agents interact with tools, external APIs, and LLMs — all of which can fail. Consistent error handling:
- Provides actionable feedback to users
- Enables retry and fallback logic
- Makes debugging easier

## Tool Execution Errors

When a tool raises an exception, wrap it with context:

```python
from smolagents import Tool
from typing import Any

class SafeWebSearchTool(Tool):
    name = "web_search"
    description = "Search the web for information"
    inputs = {"query": {"type": "string", "description": "Search query"}}
    output_type = "string"

    def forward(self, query: str) -> str:
        try:
            return self._do_search(query)
        except ConnectionError as exc:
            raise RuntimeError(
                f"Web search failed for query '{query}': {exc}"
            ) from exc

    def _do_search(self, query: str) -> str:
        # actual implementation
        ...
```

## Agent Step Errors

Use `max_steps` and handle `AgentMaxStepsError`:

```python
from smolagents import CodeAgent, HfApiModel
from smolagents.agents import AgentMaxStepsError

agent = CodeAgent(
    tools=[],
    model=HfApiModel(),
    max_steps=10,
)

try:
    result = agent.run("Solve a complex problem")
except AgentMaxStepsError as exc:
    print(f"Agent exhausted step budget: {exc}")
    # Retrieve partial output from agent.logs
```

## LLM API Errors

Handle transient failures with exponential backoff:

```python
import time
from typing import TypeVar, Callable

T = TypeVar("T")


def retry_with_backoff(
    fn: Callable[[], T],
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> T:
    """Retry a callable with exponential backoff on transient errors."""
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"Attempt {attempt + 1} failed ({exc}), retrying in {delay}s")
            time.sleep(delay)
    raise RuntimeError("Unreachable")  # satisfies type checker
```

## Logging Agent Actions

Inspect `agent.logs` after a run to understand what happened:

```python
for step in agent.logs:
    if hasattr(step, "error"):
        print(f"Step {step.step_number} error: {step.error}")
    elif hasattr(step, "tool_calls"):
        for call in step.tool_calls:
            print(f"  Called {call.name}({call.arguments})")
```

## Best Practices

1. **Always provide `max_steps`** — prevents runaway agents from consuming quota.
2. **Validate tool inputs early** — raise `ValueError` with a clear message before expensive operations.
3. **Return error strings, not raise** — tools may return `"Error: <reason>"` so the agent can attempt recovery.
4. **Log at the tool level** — use Python's `logging` module in tools, not `print`.
5. **Test with synthetic failures** — inject errors in unit tests to verify agent resilience.
