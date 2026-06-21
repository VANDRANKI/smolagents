#!/usr/bin/env python
# coding=utf-8

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import argparse
import os

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table

from smolagents import (
    CodeAgent,
    InferenceClientModel,
    LiteLLMModel,
    Model,
    OpenAIModel,
    Tool,
    ToolCallingAgent,
    TransformersModel,
)
from smolagents.default_tools import TOOL_MAPPING


console = Console()

leopard_prompt = "How many seconds would it take for a leopard at full speed to run through Pont des Arts?"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for the smolagents CLI.

    Returns:
        Parsed argument namespace.  When no ``prompt`` is supplied the CLI
        enters interactive mode.
    """
    parser = argparse.ArgumentParser(description="Run a CodeAgent with all specified parameters")
    parser.add_argument(
        "prompt",
        type=str,
        nargs="?",
        default=None,
        help="The prompt to run with the agent. If no prompt is provided, interactive mode will be launched to guide user through agent setup",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="InferenceClientModel",
        help="The model type to use (e.g., InferenceClientModel, OpenAIModel, LiteLLMModel, TransformersModel)",
    )
    parser.add_argument(
        "--action-type",
        type=str,
        default="code",
        help="The action type to use (e.g., code, tool_calling)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-Next-80B-A3B-Thinking",
        help="The model ID to use for the specified model type",
    )
    parser.add_argument(
        "--imports",
        nargs="*",  # accepts zero or more arguments
        default=[],
        help="Space-separated list of imports to authorize (e.g., 'numpy pandas')",
    )
    parser.add_argument(
        "--tools",
        nargs="*",
        default=["web_search"],
        help="Space-separated list of tools that the agent can use (e.g., 'tool1 tool2 tool3')",
    )
    parser.add_argument(
        "--verbosity-level",
        type=int,
        default=1,
        help="The verbosity level, as an int in [0, 1, 2].",
    )
    group = parser.add_argument_group("api options", "Options for API-based model types")
    group.add_argument(
        "--provider",
        type=str,
        default=None,
        help="The inference provider to use for the model",
    )
    group.add_argument(
        "--api-base",
        type=str,
        help="The base URL for the model",
    )
    group.add_argument(
        "--api-key",
        type=str,
        help="The API key for the model",
    )
    return parser.parse_args()


def interactive_mode() -> tuple[str, list[str], str, str, str | None, str | None, str | None, list[str], str]:
    """Run the CLI in interactive mode, prompting the user for all settings.

    Returns:
        A tuple of
        ``(prompt, tools, model_type, model_id, provider, api_base, api_key,
        imports, action_type)`` collected from the user.
    """
    console.print(
        Panel.fit(
            "[bold magenta]\U0001f916 SmolaGents CLI[/]\n[dim]Intelligent agents at your service[/]", border_style="magenta"
        )
    )

    console.print("\n[bold yellow]Welcome to smolagents![/] Let's set up your agent step by step.\n")

    # Get user input step by step
    console.print(Rule("[bold yellow]⚙️  Configuration", style="bold yellow"))

    # Get agent action type
    action_type = Prompt.ask(
        "[bold white]What action type would you like to use? 'code' or 'tool_calling'?[/]",
        default="code",
        choices=["code", "tool_calling"],
    )

    # Show available tools
    tools_table = Table(title="[bold yellow]\U0001f6e0️  Available Tools", show_header=True, header_style="bold yellow")
    tools_table.add_column("Tool Name", style="bold yellow")
    tools_table.add_column("Description", style="white")

    for tool_name, tool_class in TOOL_MAPPING.items():
        # Get description from the tool class if available
        try:
            tool_instance = tool_class()
            description = getattr(tool_instance, "description", "No description available")
        except Exception:
            description = "Built-in tool"
        tools_table.add_row(tool_name, description)

    console.print(tools_table)
    console.print(
        "\n[dim]You can also use HuggingFace Spaces by providing the full path (e.g., 'username/spacename')[/]"
    )

    console.print("[dim]Enter tool names separated by spaces (e.g., 'web_search python_interpreter')[/]")
    tools_input = Prompt.ask("[bold white]Select tools for your agent[/]", default="web_search")
    tools = tools_input.split()

    # Get model configuration
    console.print("\n[bold yellow]Model Configuration:[/]")
    model_type = Prompt.ask(
        "[bold]Model type[/]",
        default="InferenceClientModel",
        choices=["InferenceClientModel", "OpenAIServerModel", "LiteLLMModel", "TransformersModel"],
    )

    model_id = Prompt.ask("[bold white]Model ID[/]", default="Qwen/Qwen2.5-Coder-32B-Instruct")

    # Optional configurations
    provider = None
    api_base = None
    api_key = None
    imports: list[str] = []
    # NOTE: action_type is already set from the prompt above; do not reset it here.

    if Confirm.ask("\n[bold white]Configure advanced options?[/]", default=False):
        if model_type in ["InferenceClientModel", "OpenAIServerModel", "LiteLLMModel"]:
            provider = Prompt.ask("[bold white]Provider[/]", default="")
            api_base = Prompt.ask("[bold white]API Base URL[/]", default="")
            api_key = Prompt.ask("[bold white]API Key[/]", default="", password=True)

        imports_input = Prompt.ask("[bold white]Additional imports (space-separated)[/]", default="")
        if imports_input:
            imports = imports_input.split()

    # Get prompt
    prompt = Prompt.ask(
        "[bold white]Now the final step; what task would you like the agent to perform?[/]", default=leopard_prompt
    )

    return prompt, tools, model_type, model_id, provider, api_base, api_key, imports, action_type


def load_model(
    model_type: str,
    model_id: str,
    api_base: str | None = None,
    api_key: str | None = None,
    provider: str | None = None,
) -> Model:
    """Instantiate the requested model class.

    Args:
        model_type: One of ``"OpenAIModel"``, ``"LiteLLMModel"``,
            ``"TransformersModel"``, or ``"InferenceClientModel"``.
        model_id: Model identifier string (e.g. a HuggingFace repo ID or an
            OpenAI model name).
        api_base: Optional base URL for API-backed models.
        api_key: Optional API key.  Falls back to environment variables when
            ``None`` (``FIREWORKS_API_KEY`` for OpenAI, ``HF_API_KEY`` for
            InferenceClient).
        provider: Optional inference provider name (used by
            ``InferenceClientModel``).

    Returns:
        An initialised :class:`~smolagents.Model` instance.

    Raises:
        ValueError: If ``model_type`` is not one of the supported values.
    """
    if model_type == "OpenAIModel":
        return OpenAIModel(
            api_key=api_key or os.getenv("FIREWORKS_API_KEY"),
            api_base=api_base or "https://api.fireworks.ai/inference/v1",
            model_id=model_id,
        )
    elif model_type == "LiteLLMModel":
        return LiteLLMModel(
            model_id=model_id,
            api_key=api_key,
            api_base=api_base,
        )
    elif model_type == "TransformersModel":
        return TransformersModel(model_id=model_id, device_map="auto")
    elif model_type == "InferenceClientModel":
        return InferenceClientModel(
            model_id=model_id,
            token=api_key or os.getenv("HF_API_KEY"),
            provider=provider,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def run_smolagent(
    prompt: str,
    tools: list[str],
    model_type: str,
    model_id: str,
    api_base: str | None = None,
    api_key: str | None = None,
    imports: list[str] | None = None,
    provider: str | None = None,
    action_type: str = "code",
) -> None:
    """Build and run a smolagents agent from plain configuration values.

    Loads environment variables from a ``.env`` file, instantiates the
    requested model and tools, builds either a
    :class:`~smolagents.CodeAgent` or a
    :class:`~smolagents.ToolCallingAgent`, and calls
    :meth:`~smolagents.MultiStepAgent.run` with ``prompt``.

    Args:
        prompt: The task description to pass to the agent.
        tools: List of tool names.  Each entry is either a key in
            :data:`~smolagents.default_tools.TOOL_MAPPING` or a HuggingFace
            Space path (``"owner/space-name"``).
        model_type: Model class name; see :func:`load_model` for accepted
            values.
        model_id: Model identifier passed to the model constructor.
        api_base: Optional API base URL.
        api_key: Optional API key.
        imports: Additional Python module names that the
            :class:`~smolagents.CodeAgent` is allowed to import inside
            generated code.  Ignored when ``action_type="tool_calling"``.
        provider: Optional inference provider name.
        action_type: Either ``"code"`` (default) for
            :class:`~smolagents.CodeAgent` or ``"tool_calling"`` for
            :class:`~smolagents.ToolCallingAgent`.

    Raises:
        ValueError: If a tool name is not recognised and is not a valid
            HuggingFace Space path, or if ``action_type`` is unsupported.
    """
    load_dotenv()

    model = load_model(model_type, model_id, api_base=api_base, api_key=api_key, provider=provider)

    available_tools: list[Tool] = []

    for tool_name in tools:
        if "/" in tool_name:
            space_name = tool_name.split("/")[-1].lower().replace("-", "_").replace(".", "_")
            description = f"Tool loaded from Hugging Face Space: {tool_name}"
            available_tools.append(Tool.from_space(space_id=tool_name, name=space_name, description=description))
        else:
            if tool_name in TOOL_MAPPING:
                available_tools.append(TOOL_MAPPING[tool_name]())
            else:
                raise ValueError(f"Tool {tool_name} is not recognized either as a default tool or a Space.")

    if action_type == "code":
        agent = CodeAgent(
            tools=available_tools,
            model=model,
            additional_authorized_imports=imports,
            stream_outputs=True,
        )
    elif action_type == "tool_calling":
        agent = ToolCallingAgent(tools=available_tools, model=model, stream_outputs=True)
    else:
        raise ValueError(f"Unsupported action type: {action_type}")

    agent.run(prompt)


def main() -> None:
    """Entry point for the ``smolagents`` CLI command.

    Parses command-line arguments (see :func:`parse_arguments`).  When no
    ``prompt`` argument is supplied, drops into :func:`interactive_mode` to
    collect all settings from the user interactively.  Then delegates to
    :func:`run_smolagent`.
    """
    args = parse_arguments()

    # Check if we should run in interactive mode
    # Interactive mode is triggered when no prompt is provided
    if args.prompt is None:
        prompt, tools, model_type, model_id, provider, api_base, api_key, imports, action_type = interactive_mode()
    else:
        prompt = args.prompt
        tools = args.tools
        model_type = args.model_type
        model_id = args.model_id
        provider = args.provider
        api_base = args.api_base
        api_key = args.api_key
        imports = args.imports
        action_type = args.action_type

    run_smolagent(
        prompt,
        tools,
        model_type,
        model_id,
        provider=provider,
        api_base=api_base,
        api_key=api_key,
        imports=imports,
        action_type=action_type,
    )


if __name__ == "__main__":
    main()
