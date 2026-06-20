# Debugging Agents in smolagents

Practical techniques for diagnosing and fixing common issues in smolagents
applications.

---

## Enabling verbose logging

Set the log level to `DEBUG` to see every tool call, LLM request, and
intermediate reasoning step:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Or use the `verbosity_level` argument on the agent:

```python
from smolagents import CodeAgent, HfApiModel

agent = CodeAgent(
    tools=[...],
    model=HfApiModel(),
    verbosity_level=2,   # 0=silent, 1=info, 2=debug
)
```

---

## Common errors

### `ToolCallingError: Tool 'X' returned an unexpected type`

All tool `forward()` methods must return a `str`. If your tool produces a
structured object, serialise it before returning:

```python
import json
from smolagents import Tool

class SearchTool(Tool):
    name = "search"
    description = "Search the web and return results as JSON."
    inputs = {"query": {"type": "string", "description": "Search query"}}
    output_type = "string"

    def forward(self, query: str) -> str:
        results = _do_search(query)   # returns list[dict]
        return json.dumps(results, ensure_ascii=False)
```

### `MaxStepsExceeded`

The agent exceeded the `max_steps` limit without producing a final answer.

**Diagnosis:** Enable verbose logging and check whether the agent is stuck in
a loop, calling the same tool repeatedly with the same arguments.

**Fix options:**

1. Increase `max_steps` if the task genuinely needs more iterations.
2. Add a `done` tool that the agent can call when it has enough information.
3. Strengthen the system prompt to discourage repetitive tool calls.

---

## Tool authoring checklist

- [ ] `name`: lowercase, underscore-separated (e.g. `"fetch_webpage"`).
- [ ] `description`: one clear sentence explaining what the tool does and when to use it.
- [ ] `inputs`: all parameters described with `type` and `description`.
- [ ] `output_type`: always `"string"` unless the orchestrator handles other types.
- [ ] `forward()`: returns a plain `str` and never raises uncaught exceptions.
- [ ] Unit test: at least one test for the happy path and one for an invalid input.

---

## Inspecting the agent's memory

```python
# After running the agent, inspect the full run log
for step in agent.logs:
    print(step)
```

Each step log contains:
- `tool_name` and `tool_arguments` for tool calls
- `observations` from the tool response
- `llm_output` for reasoning traces
