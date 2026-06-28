# Agent Testing Patterns

This guide covers approaches for testing smolagents tools and agent pipelines
without running live model inference on every test run.

## 1. Testing tools in isolation

Tools decorated with `@tool` are plain Python callables. Test them directly.

```python
from myproject.tools import get_weather

def test_get_weather_returns_dict():
    result = get_weather(location="Paris")
    assert isinstance(result, dict)
    assert "temperature" in result
```

## 2. Mocking external APIs inside tools

Use `unittest.mock.patch` to avoid real HTTP calls in unit tests.

```python
from unittest.mock import patch
from myproject.tools import search_web

def test_search_web_parses_results():
    fake_response = [{"title": "Example", "url": "https://example.com"}]
    with patch("myproject.tools.requests.get") as mock_get:
        mock_get.return_value.json.return_value = fake_response
        results = search_web(query="test")
    assert results[0]["title"] == "Example"
```

## 3. Stubbing the model in CodeAgent

Replace the model with a stub that returns deterministic code.

```python
from smolagents import CodeAgent
from unittest.mock import MagicMock

def test_agent_calls_tool():
    mock_model = MagicMock()
    # Return a code snippet that calls the tool, then terminates.
    mock_model.return_value.content = 'result = my_tool(x=1)\nfinal_answer(result)'
    agent = CodeAgent(tools=[my_tool], model=mock_model)
    output = agent.run("call my tool with x=1")
    assert output is not None
```

## 4. Integration test with a cheap real model

For integration tests use a fast, inexpensive model and a simple verifiable task.

```python
import pytest
from smolagents import CodeAgent, HfApiModel
from myproject.tools import calculator

@pytest.mark.integration
def test_agent_arithmetic():
    model = HfApiModel(model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    agent = CodeAgent(tools=[calculator], model=model)
    result = agent.run("What is 7 multiplied by 8?")
    assert "56" in str(result)
```

Run integration tests separately to avoid slowing the unit-test suite:

```bash
pytest -m integration tests/
```

## 5. Snapshot testing for tool output

Freeze expected tool output to catch regressions.

```python
import json
from pathlib import Path

SNAPSHOT_DIR = Path("tests/snapshots")

def test_tool_output_snapshot():
    result = my_tool(param="value")
    snapshot_path = SNAPSHOT_DIR / "my_tool_value.json"
    if not snapshot_path.exists():
        snapshot_path.write_text(json.dumps(result, indent=2))
    expected = json.loads(snapshot_path.read_text())
    assert result == expected
```

## Anti-patterns

| Anti-pattern | Why it hurts | Fix |
|---|---|---|
| Calling GPT-4 in unit tests | Slow, expensive, non-deterministic | Stub the model |
| Testing only the happy path | Misses edge cases | Test invalid inputs too |
| No isolation between tool tests | Hidden side-effects | Use `patch` or temp dirs |
