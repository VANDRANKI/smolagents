"""Output parsing utilities for smolagents tool results.

These parsers extract structured data from LLM output strings before
passing results to the next step in the agent loop.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional


def extract_code_block(text: str, language: str = "python") -> Optional[str]:
    """Extract the first fenced code block of the given language from *text*.

    Args:
        text: The raw LLM output string.
        language: The language tag to match (e.g. ``'python'``, ``'bash'``).
            Case-insensitive.

    Returns:
        The code content inside the code fence, or ``None`` if no matching
        block is found.
    """
    pattern = rf"```{re.escape(language)}\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return match.group(1).rstrip() if match else None


def extract_json_object(text: str) -> Optional[Any]:
    """Extract and parse the first JSON object or array from *text*.

    Handles output where the LLM wraps JSON inside prose or markdown.

    Args:
        text: The raw LLM output string.

    Returns:
        The parsed Python value (dict, list, etc.), or ``None`` if no
        valid JSON is found.
    """
    match = re.search(r"(\{.*?\}|\[.*?\])", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError:
        return None


def extract_final_answer(text: str) -> Optional[str]:
    """Extract the value after a 'Final Answer:' marker in *text*.

    Many ReAct-style prompts terminate with a ``Final Answer:`` line.
    This parser extracts everything after that marker.

    Args:
        text: The raw LLM output string.

    Returns:
        The text following the ``Final Answer:`` marker, stripped of
        leading/trailing whitespace, or ``None`` if no marker is found.
    """
    match = re.search(r"Final Answer:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None
