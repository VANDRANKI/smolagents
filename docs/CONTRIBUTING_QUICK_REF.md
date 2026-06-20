# smolagents Contributor Quick Reference

## Setup

```bash
git clone https://github.com/huggingface/smolagents.git
cd smolagents
pip install -e ".[dev]"
```

## Running tests

```bash
# Run all tests
pytest tests/ -v

# Run a specific test
pytest tests/test_tools.py -v -k test_search_tool

# Run with a real LM (requires API key)
HF_TOKEN=... pytest tests/test_e2e.py -v
```

## Writing a new tool — checklist

- [ ] `name`: lowercase, underscore-separated.
- [ ] `description`: one clear sentence.
- [ ] All `inputs` have `type` and `description`.
- [ ] `output_type = "string"`.
- [ ] `forward()` returns a plain `str`.
- [ ] `forward()` never raises uncaught exceptions; catch and return an error
  string instead.
- [ ] Unit test in `tests/test_tools.py`.

## Formatting

```bash
ruff format .
ruff check . --fix
```

## Type checking

```bash
mypy src/smolagents/ --ignore-missing-imports
```
