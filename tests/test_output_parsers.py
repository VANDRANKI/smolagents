"""Unit tests for smolagents.utils.output_parsers."""

import pytest
from smolagents.utils.output_parsers import (
    extract_code_block,
    extract_json_object,
    extract_final_answer,
)


class TestExtractCodeBlock:
    def test_extracts_python_block(self):
        text = "Here is the code:\n```python\nprint('hello')\n```\nDone."
        assert extract_code_block(text) == "print('hello')"

    def test_returns_none_when_no_block(self):
        assert extract_code_block("No code here") is None

    def test_extracts_bash_block(self):
        text = "```bash\necho hello\n```"
        assert extract_code_block(text, language="bash") == "echo hello"

    def test_case_insensitive_language_tag(self):
        text = "```Python\nresult = 1 + 1\n```"
        assert extract_code_block(text, language="python") == "result = 1 + 1"

    def test_extracts_first_block_when_multiple(self):
        text = "```python\nfirst()\n```\n```python\nsecond()\n```"
        assert extract_code_block(text) == "first()"


class TestExtractJsonObject:
    def test_extracts_dict(self):
        text = 'The answer is {"key": "value", "n": 42}.'
        result = extract_json_object(text)
        assert result == {"key": "value", "n": 42}

    def test_extracts_array(self):
        text = "Result: [1, 2, 3]"
        assert extract_json_object(text) == [1, 2, 3]

    def test_returns_none_for_invalid_json(self):
        assert extract_json_object("no json here") is None

    def test_returns_none_for_malformed_json(self):
        assert extract_json_object("{key: value}") is None


class TestExtractFinalAnswer:
    def test_extracts_answer(self):
        text = "I searched...\nFinal Answer: Paris"
        assert extract_final_answer(text) == "Paris"

    def test_case_insensitive(self):
        text = "FINAL ANSWER: 42"
        assert extract_final_answer(text) == "42"

    def test_returns_none_when_no_marker(self):
        assert extract_final_answer("No answer marker here") is None

    def test_strips_whitespace(self):
        text = "Final Answer:   lots of spaces   "
        assert extract_final_answer(text) == "lots of spaces"
