#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from dataclasses import dataclass, field
from enum import IntEnum

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from smolagents.utils import sanitize_for_rich


__all__ = ["AgentLogger", "LogLevel", "Monitor", "TokenUsage", "Timing"]


@dataclass
class TokenUsage:
    """Token usage counters for a single agent step or a complete run.

    Attributes:
        input_tokens (int): Number of tokens sent to the model (prompt tokens).
        output_tokens (int): Number of tokens produced by the model (completion tokens).
        total_tokens (int): Sum of ``input_tokens`` and ``output_tokens``,
            computed automatically in ``__post_init__``.
    """

    input_tokens: int
    output_tokens: int
    total_tokens: int = field(init=False)

    def __post_init__(self) -> None:
        self.total_tokens = self.input_tokens + self.output_tokens

    def dict(self) -> dict:
        """Return a plain-dict representation suitable for JSON serialisation."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class Timing:
    """Wall-clock timing information for a single agent step or a complete run.

    Attributes:
        start_time (float): Unix timestamp (seconds) recorded at step start.
        end_time (float | None): Unix timestamp (seconds) recorded at step end,
            or ``None`` if the step has not yet finished.
    """

    start_time: float
    end_time: float | None = None

    @property
    def duration(self) -> float | None:
        """Elapsed time in seconds, or ``None`` if the step is still running."""
        return None if self.end_time is None else self.end_time - self.start_time

    def dict(self) -> dict:
        """Return a plain-dict representation suitable for JSON serialisation."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
        }

    def __repr__(self) -> str:
        return f"Timing(start_time={self.start_time}, end_time={self.end_time}, duration={self.duration})"


class Monitor:
    """Tracks cumulative token usage and step durations across an agent run.

    One ``Monitor`` instance is attached to an agent at construction time and
    is updated after every :class:`~smolagents.memory.ActionStep` completes.
    It does *not* directly call any model — it only collects metrics reported
    by the model client.

    Args:
        tracked_model: The model client (or wrapper) whose calls are being
            monitored.  Used for informational purposes only; the monitor does
            not call the model itself.
        logger (AgentLogger): Logger used to print per-step summaries.

    Attributes:
        step_durations (list[float]): Wall-clock durations (seconds) of each
            completed step, in order.
        total_input_token_count (int): Running total of input tokens sent to
            the model across all steps in the current run.
        total_output_token_count (int): Running total of output tokens received
            from the model across all steps in the current run.
    """

    def __init__(self, tracked_model, logger: "AgentLogger") -> None:
        self.step_durations: list[float] = []
        self.tracked_model = tracked_model
        self.logger = logger
        self.total_input_token_count: int = 0
        self.total_output_token_count: int = 0

    def get_total_token_counts(self) -> TokenUsage:
        """Return a :class:`TokenUsage` snapshot of the current cumulative totals."""
        return TokenUsage(
            input_tokens=self.total_input_token_count,
            output_tokens=self.total_output_token_count,
        )

    def reset(self) -> None:
        """Reset all counters and duration history to their initial state."""
        self.step_durations = []
        self.total_input_token_count = 0
        self.total_output_token_count = 0

    def update_metrics(self, step_log) -> None:
        """Update running metrics after a step completes and log a summary line.

        Args:
            step_log (ActionStep): The completed step whose ``timing`` and
                optional ``token_usage`` attributes are read.
        """
        step_duration = step_log.timing.duration
        self.step_durations.append(step_duration)
        console_outputs = f"[Step {len(self.step_durations)}: Duration {step_duration:.2f} seconds"

        if step_log.token_usage is not None:
            self.total_input_token_count += step_log.token_usage.input_tokens
            self.total_output_token_count += step_log.token_usage.output_tokens
            console_outputs += (
                f"| Input tokens: {self.total_input_token_count:,} | Output tokens: {self.total_output_token_count:,}"
            )
        console_outputs += "]"
        self.logger.log(Text(console_outputs, style="dim"), level=1)


class LogLevel(IntEnum):
    """Verbosity levels controlling which messages the :class:`AgentLogger` emits.

    Levels are ordered: a logger with a given level will print any message
    whose level is *less than or equal to* the logger's level.
    """

    OFF = -1    # Suppress all output.
    ERROR = 0   # Errors only (always shown unless OFF).
    INFO = 1    # Normal operational output (default).
    DEBUG = 2   # Verbose diagnostic output.


YELLOW_HEX = "#d4b702"


class AgentLogger:
    """Rich-based console logger used throughout the smolagents agent loop.

    All agent output — task panels, code blocks, step rules, model messages —
    flows through this class.  The ``level`` attribute controls how much is
    printed: set it to :attr:`LogLevel.DEBUG` for full diagnostic output, or
    :attr:`LogLevel.OFF` to silence everything (e.g. in tests).

    Args:
        level (LogLevel): Minimum verbosity level to emit.  Messages with a
            level *higher* than this value are silently discarded.
            Defaults to :attr:`LogLevel.INFO`.
        console (rich.console.Console | None): A pre-configured Rich
            ``Console`` to write to.  When ``None`` (the default) a new
            ``Console`` is created with ``highlight=False`` so that agent
            output is not accidentally colourised by Rich's auto-highlighter.

    Attributes:
        level (LogLevel): The current verbosity threshold.
        console (rich.console.Console): The underlying Rich console.
    """

    def __init__(self, level: LogLevel = LogLevel.INFO, console: Console | None = None) -> None:
        self.level = level
        if console is None:
            self.console = Console(highlight=False)
        else:
            self.console = console

    def log(self, *args, level: int | str | LogLevel = LogLevel.INFO, **kwargs) -> None:
        """Emit a Rich-renderable to the console if ``level`` is within threshold.

        Args:
            *args: Positional arguments forwarded to ``Console.print``.
            level (LogLevel | int | str): Verbosity level of this message.
                String values are looked up by name (case-insensitive).
                Defaults to :attr:`LogLevel.INFO`.
            **kwargs: Keyword arguments forwarded to ``Console.print``.
        """
        if isinstance(level, str):
            level = LogLevel[level.upper()]
        if level <= self.level:
            self.console.print(*args, **kwargs)

    def log_error(self, error_message: str) -> None:
        """Emit *error_message* in bold red at :attr:`LogLevel.ERROR` priority."""
        self.log(Text(sanitize_for_rich(error_message), style="bold red"), level=LogLevel.ERROR)

    def log_markdown(self, content: str, title: str | None = None, level=LogLevel.INFO, style=YELLOW_HEX) -> None:
        """Render *content* as a Markdown/code block, with an optional titled rule above it.

        Args:
            content (str): Raw text to render (treated as Markdown).
            title (str | None): If provided, a styled rule with this title is
                printed above the content block.
            level (LogLevel): Verbosity level.  Defaults to :attr:`LogLevel.INFO`.
            style (str): Rich colour string applied to the rule.  Defaults to
                the agent's gold colour.
        """
        markdown_content = Syntax(
            content,
            lexer="markdown",
            theme="github-dark",
            word_wrap=True,
        )
        if title:
            self.log(
                Group(
                    Rule(
                        "[bold italic]" + title,
                        align="left",
                        style=style,
                    ),
                    markdown_content,
                ),
                level=level,
            )
        else:
            self.log(markdown_content, level=level)

    def log_code(self, title: str, content: str, level: int = LogLevel.INFO) -> None:
        """Render *content* as a syntax-highlighted Python block inside a panel.

        Args:
            title (str): Panel title displayed in the top-left corner.
            content (str): Python source code to display.
            level (int): Verbosity level.  Defaults to :attr:`LogLevel.INFO`.
        """
        self.log(
            Panel(
                Syntax(
                    content,
                    lexer="python",
                    theme="monokai",
                    word_wrap=True,
                ),
                title="[bold]" + title,
                title_align="left",
                box=box.HORIZONTALS,
            ),
            level=level,
        )

    def log_rule(self, title: str, level: int = LogLevel.INFO) -> None:
        """Print a full-width horizontal rule with *title*.

        Args:
            title (str): Text centred on the rule line.
            level (int): Verbosity level.  Defaults to :attr:`LogLevel.INFO`.
        """
        self.log(
            Rule(
                "[bold white]" + title,
                characters="━",
                style=YELLOW_HEX,
            ),
            level=LogLevel.INFO,
        )

    def log_task(self, content: str, subtitle: str, title: str | None = None, level: LogLevel = LogLevel.INFO) -> None:
        # Important: `content` can contain arbitrary tool logs / payloads. If we embed it
        # inside Rich markup (e.g. f"[bold]{content}"), any stray "[/...]" sequences or
        # binary-ish characters can crash Rich's markup parser. Render the content as
        # `Text` instead, and apply styling via Text/style, not markup.
        safe_content = sanitize_for_rich(content)
        safe_subtitle = sanitize_for_rich(subtitle)
        content_text = Text("\n") + Text(safe_content, style="bold") + Text("\n")
        subtitle_text = Text(safe_subtitle)
        self.log(
            Panel(
                content_text,
                title="[bold]New run" + (f" - {title}" if title else ""),
                subtitle=subtitle_text,
                border_style=YELLOW_HEX,
                subtitle_align="left",
            ),
            level=level,
        )

    def log_messages(self, messages: list[dict], level: LogLevel = LogLevel.DEBUG) -> None:
        """Print raw model messages as formatted JSON for debugging.

        Args:
            messages (list[dict]): List of :class:`~smolagents.models.ChatMessage`
                objects to serialise and display.
            level (LogLevel): Verbosity level.  Defaults to :attr:`LogLevel.DEBUG`.
        """
        messages_as_string = "\n".join([json.dumps(message.dict(), indent=4) for message in messages])
        self.log(
            Syntax(
                messages_as_string,
                lexer="markdown",
                theme="github-dark",
                word_wrap=True,
            ),
            level=level,
        )

    def visualize_agent_tree(self, agent) -> None:
        """Print a Rich tree visualisation of *agent* and its managed sub-agents.

        Args:
            agent: A smolagents agent instance (e.g. ``CodeAgent``,
                ``ToolCallingAgent``) whose ``tools`` and ``managed_agents``
                attributes are inspected recursively.
        """
        def create_tools_section(tools_dict):
            table = Table(show_header=True, header_style="bold")
            table.add_column("Name", style="#1E90FF")
            table.add_column("Description")
            table.add_column("Arguments")

            for name, tool in tools_dict.items():
                args = [
                    f"{arg_name} (`{info.get('type', 'Any')}`{', optional' if info.get('optional') else ''}): {info.get('description', '')}"
                    for arg_name, info in getattr(tool, "inputs", {}).items()
                ]
                table.add_row(name, getattr(tool, "description", str(tool)), "\n".join(args))

            return Group("\U0001f6e0️ [italic #1E90FF]Tools:[/italic #1E90FF]", table)

        def get_agent_headline(agent, name: str | None = None):
            name_headline = f"{name} | " if name else ""
            return f"[bold {YELLOW_HEX}]{name_headline}{agent.__class__.__name__} | {agent.model.model_id}"

        def build_agent_tree(parent_tree, agent_obj):
            """Recursively builds the agent tree."""
            parent_tree.add(create_tools_section(agent_obj.tools))

            if agent_obj.managed_agents:
                agents_branch = parent_tree.add("\U0001f916 [italic #1E90FF]Managed agents:")
                for name, managed_agent in agent_obj.managed_agents.items():
                    agent_tree = agents_branch.add(get_agent_headline(managed_agent, name))
                    if managed_agent.__class__.__name__ == "CodeAgent":
                        agent_tree.add(
                            f"✅ [italic #1E90FF]Authorized imports:[/italic #1E90FF] {managed_agent.additional_authorized_imports}"
                        )
                    agent_tree.add(f"\U0001f4dd [italic #1E90FF]Description:[/italic #1E90FF] {managed_agent.description}")
                    build_agent_tree(agent_tree, managed_agent)

        main_tree = Tree(get_agent_headline(agent))
        if agent.__class__.__name__ == "CodeAgent":
            main_tree.add(
                f"✅ [italic #1E90FF]Authorized imports:[/italic #1E90FF] {agent.additional_authorized_imports}"
            )
        build_agent_tree(main_tree, agent)
        self.console.print(main_tree)
