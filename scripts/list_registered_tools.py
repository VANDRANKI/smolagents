#!/usr/bin/env python3
"""List all tools registered on a smolagents agent.

Useful for quickly auditing the tool surface of an agent without running it.

Usage::

    python scripts/list_registered_tools.py --module my_agent --attr agent
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any


def list_tools(agent: Any) -> list[dict[str, str]]:
    """Extract tool metadata from a smolagents agent.

    Args:
        agent: A smolagents ``Agent`` instance.

    Returns:
        List of dicts with ``name``, ``description``, and ``output_type`` keys.
    """
    toolbox = getattr(agent, "toolbox", None) or getattr(agent, "tools", [])
    if hasattr(toolbox, "tools"):
        tools_iter = toolbox.tools.values()
    elif isinstance(toolbox, (list, tuple)):
        tools_iter = toolbox
    else:
        tools_iter = []

    result: list[dict[str, str]] = []
    for tool in tools_iter:
        result.append(
            {
                "name": getattr(tool, "name", str(tool)),
                "description": getattr(tool, "description", ""),
                "output_type": getattr(tool, "output_type", "unknown"),
            }
        )
    return result


def main() -> None:
    """Entry point for the tool lister."""
    parser = argparse.ArgumentParser(
        description="List tools registered on a smolagents agent."
    )
    parser.add_argument("--module", required=True, help="Module containing the agent.")
    parser.add_argument("--attr", default="agent", help="Attribute name of the agent.")
    args = parser.parse_args()

    try:
        mod = importlib.import_module(args.module)
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    agent = getattr(mod, args.attr, None)
    if agent is None:
        print(f"ERROR: '{args.attr}' not found in '{args.module}'.", file=sys.stderr)
        sys.exit(1)

    tools = list_tools(agent)
    if not tools:
        print("No tools found.")
        return

    max_name = max(len(t["name"]) for t in tools)
    header = f"{'Name':<{max_name}}  Output type  Description"
    print(header)
    print("-" * len(header))
    for t in tools:
        desc = t["description"][:60] + "..." if len(t["description"]) > 60 else t["description"]
        print(f"{t['name']:<{max_name}}  {t['output_type']:<11}  {desc}")


if __name__ == "__main__":
    main()
