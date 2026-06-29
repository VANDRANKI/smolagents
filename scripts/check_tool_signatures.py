#!/usr/bin/env python3
"""Check that smolagents Tool subclasses have well-formed signatures.

Imports all Tool subclasses from src/smolagents/tools.py, verifies:
  - description is non-empty
  - forward() accepts at least one positional argument
  - inputs dict is defined (used to build the JSON schema)
  - output_type is set

Usage:
    python scripts/check_tool_signatures.py
    python scripts/check_tool_signatures.py --verbose
"""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

SRC_DIR = Path("src")


def discover_tool_classes():
    """Import smolagents and return all non-abstract Tool subclasses."""
    sys.path.insert(0, str(SRC_DIR))
    try:
        import smolagents  # noqa: F401
        from smolagents.tools import Tool
    except ImportError as exc:
        print(f"ERROR: Could not import smolagents: {exc}", file=sys.stderr)
        sys.exit(1)

    subclasses = []
    for name, obj in inspect.getmembers(sys.modules.get("smolagents.tools", object()), inspect.isclass):
        if issubclass(obj, Tool) and obj is not Tool and not inspect.isabstract(obj):
            subclasses.append((name, obj))
    return subclasses


def check_tool(name: str, cls) -> list[str]:
    """Return a list of issues with the Tool subclass."""
    issues: list[str] = []

    description = getattr(cls, "description", None)
    if not description or not description.strip():
        issues.append("missing or empty 'description' class attribute")

    inputs = getattr(cls, "inputs", None)
    if not isinstance(inputs, dict) or not inputs:
        issues.append("'inputs' must be a non-empty dict")

    output_type = getattr(cls, "output_type", None)
    if not output_type:
        issues.append("missing 'output_type' class attribute")

    forward = getattr(cls, "forward", None)
    if forward is None:
        issues.append("missing forward() method")
    else:
        sig = inspect.signature(forward)
        params = [p for p in sig.parameters if p != "self"]
        if not params:
            issues.append("forward() has no parameters beyond 'self'")

    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    tool_classes = discover_tool_classes()
    if not tool_classes:
        print("No Tool subclasses found.")
        return 0

    all_issues: dict[str, list[str]] = {}
    for name, cls in sorted(tool_classes):
        issues = check_tool(name, cls)
        if issues:
            all_issues[name] = issues
        elif args.verbose:
            print(f"OK  {name}")

    if all_issues:
        print(f"\nSignature issues in {len(all_issues)}/{len(tool_classes)} tool(s):\n")
        for tool_name, issues in sorted(all_issues.items()):
            for issue in issues:
                print(f"  {tool_name}: {issue}")
        return 1

    print(f"All {len(tool_classes)} Tool subclasses passed signature checks.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
