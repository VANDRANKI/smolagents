"""Example: custom Tool with input validation and proper type annotations.

Run with:
  .venv/bin/python examples/custom_tool_with_validation.py
"""

from typing import Optional
from smolagents import Tool


class TextSummarizerTool(Tool):
    """Summarizes a block of text to a target word count."""

    name = "text_summarizer"
    description = (
        "Summarizes a long piece of text to a shorter version. "
        "Use this when you need to condense information from a document."
    )
    inputs = {
        "text": {
            "type": "string",
            "description": "The text to summarize.",
        },
        "target_words": {
            "type": "integer",
            "description": "Approximate target word count for the summary.",
            "nullable": True,
        },
    }
    output_type = "string"

    def forward(self, text: str, target_words: Optional[int] = None) -> str:
        """Summarize the input text.

        Args:
            text: The source text to summarize.
            target_words: Desired summary length in words. Defaults to 10% of input.

        Returns:
            A shorter summary string.

        Raises:
            ValueError: If text is empty.
        """
        if not text.strip():
            raise ValueError("Cannot summarize empty text.")

        words = text.split()
        target = target_words or max(10, len(words) // 10)

        # Naive extractive summary: take first N words
        summary_words = words[:target]
        summary = " ".join(summary_words)
        if len(words) > target:
            summary += "..."
        return summary


if __name__ == "__main__":
    tool = TextSummarizerTool()

    sample = (
        "LangChain is a framework for developing applications powered by "
        "large language models. It enables applications that are context-aware "
        "and reason about how to respond, using various components like chains, "
        "agents, and memory. The framework supports Python and JavaScript."
    )

    print("Original:", sample[:80] + "...")
    print("Summary (20 words):", tool.forward(text=sample, target_words=20))
    print("Summary (default):", tool.forward(text=sample))
