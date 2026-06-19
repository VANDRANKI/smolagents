# Output Parsing

smolagents provides utilities for extracting structured content from raw LLM outputs.

## Extracting code blocks

```python
from smolagents.utils.output_parsers import extract_code_block

llm_output = """
Here is the solution:
```python
def greet(name: str) -> str:
    return f"Hello, {name}!"
```
"""

code = extract_code_block(llm_output)
# code == "def greet(name: str) -> str:\n    return f\"Hello, {name}!\""
```

## Extracting JSON

```python
from smolagents.utils.output_parsers import extract_json_object

output = 'The result is {"status": "ok", "count": 3}.'
data = extract_json_object(output)
# data == {"status": "ok", "count": 3}
```

## Extracting final answers

Many ReAct-style prompts end with `Final Answer:`:

```python
from smolagents.utils.output_parsers import extract_final_answer

output = "I looked it up.\nFinal Answer: The Eiffel Tower is in Paris."
answer = extract_final_answer(output)
# answer == "The Eiffel Tower is in Paris."
```
