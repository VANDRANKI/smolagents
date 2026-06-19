import inspect
from dataclasses import asdict, dataclass
from logging import getLogger
from typing import TYPE_CHECKING, Any, Callable, Type

from smolagents.models import ChatMessage, MessageRole, get_dict_from_nested_dataclasses
from smolagents.monitoring import AgentLogger, LogLevel, Timing, TokenUsage
from smolagents.utils import AgentError, make_json_serializable


if TYPE_CHECKING:
    import PIL.Image

    from smolagents.models import ChatMessage
    from smolagents.monitoring import AgentLogger


__all__ = ["AgentMemory"]


logger = getLogger(__name__)


@dataclass
class ToolCall:
    """Record of a single tool invocation requested by the model.

    Attributes:
        name (str): Name of the tool that was called.
        arguments (Any): Arguments passed to the tool, typically a dict
            or JSON-serializable object.
        id (str): Unique call identifier assigned by the model (mirrors the
            OpenAI ``tool_calls[].id`` field).
    """

    name: str
    arguments: Any
    id: str

    def dict(self) -> dict:
        """Return an OpenAI-compatible tool-call dict representation."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": make_json_serializable(self.arguments),
            },
        }


@dataclass
class MemoryStep:
    """Abstract base class for all memory step types.

    Subclasses represent the different kinds of events that can occur during
    an agent run: :class:`TaskStep`, :class:`ActionStep`, :class:`PlanningStep`,
    :class:`SystemPromptStep`, and :class:`FinalAnswerStep`.
    """

    def dict(self) -> dict:
        """Return a dict representation of this step."""
        return asdict(self)

    def to_messages(self, summary_mode: bool = False) -> list["ChatMessage"]:
        """Convert this step to a list of :class:`~smolagents.models.ChatMessage` objects.

        Args:
            summary_mode (bool): When ``True``, some steps omit verbose
                content to produce a compact memory view.  Defaults to ``False``.

        Returns:
            list[ChatMessage]: Messages representing this step for inclusion in
            the model's context window.

        Raises:
            NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError


@dataclass
class ActionStep(MemoryStep):
    """Records everything that happened during a single agent action step.

    An action step corresponds to one full turn of the agent loop: the model
    is called, optionally invokes a tool, and receives an observation.

    Attributes:
        step_number (int): 1-based index of this step within the current run.
        timing (Timing): Wall-clock start/end times for the step.
        model_input_messages (list[ChatMessage] | None): Messages sent to the
            model as the prompt for this step.
        tool_calls (list[ToolCall] | None): Tool calls requested by the model,
            if any.
        error (AgentError | None): Error raised during tool execution or model
            parsing, if any.
        model_output_message (ChatMessage | None): The raw assistant message
            returned by the model.
        model_output (str | list[dict] | None): Parsed text or structured
            output from the model response.
        code_action (str | None): Extracted Python code block, populated by
            :class:`~smolagents.agents.CodeAgent`.
        observations (str | None): Text observation returned after tool
            execution or code execution.
        observations_images (list[PIL.Image.Image] | None): Image observations
            returned by visual tools.
        action_output (Any): Return value of the executed tool or code block.
        token_usage (TokenUsage | None): Token counts for this step's model
            call.
        is_final_answer (bool): ``True`` when the agent has signalled that
            this step contains the final answer to the task.
    """

    step_number: int
    timing: Timing
    model_input_messages: list["ChatMessage"] | None = None
    tool_calls: list[ToolCall] | None = None
    error: AgentError | None = None
    model_output_message: "ChatMessage | None" = None
    model_output: str | list[dict[str, Any]] | None = None
    code_action: str | None = None
    observations: str | None = None
    observations_images: list["PIL.Image.Image"] | None = None
    action_output: Any = None
    token_usage: TokenUsage | None = None
    is_final_answer: bool = False

    def dict(self) -> dict:
        # We overwrite the method to parse the tool_calls and action_output manually
        return {
            "step_number": self.step_number,
            "timing": self.timing.dict(),
            "model_input_messages": [
                make_json_serializable(get_dict_from_nested_dataclasses(msg)) for msg in self.model_input_messages
            ]
            if self.model_input_messages
            else None,
            "tool_calls": [tc.dict() for tc in self.tool_calls] if self.tool_calls else [],
            "error": self.error.dict() if self.error else None,
            "model_output_message": make_json_serializable(get_dict_from_nested_dataclasses(self.model_output_message))
            if self.model_output_message
            else None,
            "model_output": self.model_output,
            "code_action": self.code_action,
            "observations": self.observations,
            "observations_images": [image.tobytes() for image in self.observations_images]
            if self.observations_images
            else None,
            "action_output": make_json_serializable(self.action_output),
            "token_usage": asdict(self.token_usage) if self.token_usage else None,
            "is_final_answer": self.is_final_answer,
        }

    def to_messages(self, summary_mode: bool = False) -> list["ChatMessage"]:
        """Convert this action step to model-ready chat messages.

        In summary mode the raw model output text is omitted, keeping only
        tool calls, observations, and errors so that the context window is
        not inflated when replaying history.

        Args:
            summary_mode (bool): If ``True``, skip the raw model output
                message.  Defaults to ``False``.

        Returns:
            list[ChatMessage]: Between 0 and 4 messages representing:
                the model output (if any), tool calls, image observations,
                text observations, and errors.
        """
        messages = []
        if self.model_output is not None and not summary_mode:
            messages.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.model_output.strip()}])
            )

        if self.tool_calls is not None:
            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL_CALL,
                    content=[
                        {
                            "type": "text",
                            "text": "Calling tools:\n" + str([tc.dict() for tc in self.tool_calls]),
                        }
                    ],
                )
            )

        if self.observations_images:
            messages.append(
                ChatMessage(
                    role=MessageRole.USER,
                    content=[
                        {
                            "type": "image",
                            "image": image,
                        }
                        for image in self.observations_images
                    ],
                )
            )

        if self.observations is not None:
            messages.append(
                ChatMessage(
                    role=MessageRole.TOOL_RESPONSE,
                    content=[
                        {
                            "type": "text",
                            "text": f"Observation:\n{self.observations}",
                        }
                    ],
                )
            )
        if self.error is not None:
            error_message = (
                "Error:\n"
                + str(self.error)
                + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
            )
            message_content = f"Call id: {self.tool_calls[0].id}\n" if self.tool_calls else ""
            message_content += error_message
            messages.append(
                ChatMessage(role=MessageRole.TOOL_RESPONSE, content=[{"type": "text", "text": message_content}])
            )

        return messages


@dataclass
class PlanningStep(MemoryStep):
    """Records the output of an optional planning call made before action steps.

    Some agent variants call the model once at the start of a task (or
    periodically) to produce a high-level plan before executing individual
    tool-use steps.  This step captures that planning exchange.

    Attributes:
        model_input_messages (list[ChatMessage]): Prompt messages sent to the
            model for planning.
        model_output_message (ChatMessage): Raw assistant message returned by
            the model.
        plan (str): The extracted plain-text plan produced by the model.
        timing (Timing): Wall-clock start/end times for the planning call.
        token_usage (TokenUsage | None): Token counts for the planning call,
            or ``None`` if not tracked.
    """

    model_input_messages: list["ChatMessage"]
    model_output_message: "ChatMessage"
    plan: str
    timing: Timing
    token_usage: TokenUsage | None = None

    def dict(self) -> dict:
        return {
            "model_input_messages": [
                make_json_serializable(get_dict_from_nested_dataclasses(msg)) for msg in self.model_input_messages
            ],
            "model_output_message": make_json_serializable(
                get_dict_from_nested_dataclasses(self.model_output_message)
            ),
            "plan": self.plan,
            "timing": self.timing.dict(),
            "token_usage": asdict(self.token_usage) if self.token_usage else None,
        }

    def to_messages(self, summary_mode: bool = False) -> list["ChatMessage"]:
        """Convert this planning step to model-ready chat messages.

        In summary mode, planning steps are excluded from the history to
        prevent the context window from growing with repeated plan text.

        Args:
            summary_mode (bool): If ``True``, return an empty list.
                Defaults to ``False``.

        Returns:
            list[ChatMessage]: Two messages (plan + follow-up instruction) in
            normal mode, or an empty list in summary mode.
        """
        if summary_mode:
            return []
        return [
            ChatMessage(role=MessageRole.ASSISTANT, content=[{"type": "text", "text": self.plan.strip()}]),
            ChatMessage(
                role=MessageRole.USER, content=[{"type": "text", "text": "Now proceed and carry out this plan."}]
            ),
            # This second message creates a role change to prevent models from simply continuing the plan message
        ]


@dataclass
class TaskStep(MemoryStep):
    """Records the user task that initiated an agent run.

    Attributes:
        task (str): The natural-language task or question given to the agent.
        task_images (list[PIL.Image.Image] | None): Optional images provided
            alongside the text task (for multi-modal agents).
    """

    task: str
    task_images: list["PIL.Image.Image"] | None = None

    def to_messages(self, summary_mode: bool = False) -> list["ChatMessage"]:
        """Convert the task to a user chat message.

        Args:
            summary_mode (bool): Unused; task steps are always included.
                Defaults to ``False``.

        Returns:
            list[ChatMessage]: A single user message containing the task text
            and any attached images.
        """
        content = [{"type": "text", "text": f"New task:\n{self.task}"}]
        if self.task_images:
            content.extend([{"type": "image", "image": image} for image in self.task_images])

        return [ChatMessage(role=MessageRole.USER, content=content)]


@dataclass
class SystemPromptStep(MemoryStep):
    """Records the system prompt injected at the start of every agent context.

    Attributes:
        system_prompt (str): The full system prompt text given to the model.
    """

    system_prompt: str

    def to_messages(self, summary_mode: bool = False) -> list["ChatMessage"]:
        """Convert the system prompt to a system chat message.

        Args:
            summary_mode (bool): If ``True``, return an empty list so that the
                system prompt is omitted from summary-mode context.
                Defaults to ``False``.

        Returns:
            list[ChatMessage]: A single system message, or an empty list in
            summary mode.
        """
        if summary_mode:
            return []
        return [ChatMessage(role=MessageRole.SYSTEM, content=[{"type": "text", "text": self.system_prompt}])]


@dataclass
class FinalAnswerStep(MemoryStep):
    """Records the final answer produced at the end of an agent run.

    Attributes:
        output (Any): The final answer value returned to the caller.  May be
            a string, image, or any other type supported by the agent.
    """

    output: Any


class AgentMemory:
    """Memory for the agent, containing the system prompt and all steps taken by the agent.

    This class is used to store the agent's steps, including tasks, actions, and planning steps.
    It allows for resetting the memory, retrieving succinct or full step information, and replaying the agent's steps.

    Args:
        system_prompt (`str`): System prompt for the agent, which sets the context and instructions for the agent's behavior.

    **Attributes**:
        - **system_prompt** (`SystemPromptStep`) -- System prompt step for the agent.
        - **steps** (`list[TaskStep | ActionStep | PlanningStep]`) -- List of steps taken by the agent, which can include tasks, actions, and planning steps.
    """

    def __init__(self, system_prompt: str) -> None:
        self.system_prompt: SystemPromptStep = SystemPromptStep(system_prompt=system_prompt)
        self.steps: list[TaskStep | ActionStep | PlanningStep] = []

    def reset(self) -> None:
        """Reset the agent's memory, clearing all steps and keeping the system prompt."""
        self.steps = []

    def get_succinct_steps(self) -> list[dict]:
        """Return a succinct representation of the agent's steps, excluding model input messages."""
        return [
            {key: value for key, value in step.dict().items() if key != "model_input_messages"} for step in self.steps
        ]

    def get_full_steps(self) -> list[dict]:
        """Return a full representation of the agent's steps, including model input messages."""
        if len(self.steps) == 0:
            return []
        return [step.dict() for step in self.steps]

    def replay(self, logger: AgentLogger, detailed: bool = False) -> None:
        """Prints a pretty replay of the agent's steps.

        Args:
            logger (`AgentLogger`): The logger to print replay logs to.
            detailed (`bool`, default `False`): If True, also displays the memory at each step. Defaults to False.
                Careful: will increase log length exponentially. Use only for debugging.
        """
        logger.console.log("Replaying the agent's steps:")
        logger.log_markdown(title="System prompt", content=self.system_prompt.system_prompt, level=LogLevel.ERROR)
        for step in self.steps:
            if isinstance(step, TaskStep):
                logger.log_task(step.task, "", level=LogLevel.ERROR)
            elif isinstance(step, ActionStep):
                logger.log_rule(f"Step {step.step_number}", level=LogLevel.ERROR)
                if detailed and step.model_input_messages is not None:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                if step.model_output is not None:
                    logger.log_markdown(title="Agent output:", content=step.model_output, level=LogLevel.ERROR)
            elif isinstance(step, PlanningStep):
                logger.log_rule("Planning step", level=LogLevel.ERROR)
                if detailed and step.model_input_messages is not None:
                    logger.log_messages(step.model_input_messages, level=LogLevel.ERROR)
                logger.log_markdown(title="Agent output:", content=step.plan, level=LogLevel.ERROR)

    def return_full_code(self) -> str:
        """Returns all code actions from the agent's steps, concatenated as a single script."""
        return "\n\n".join(
            [step.code_action for step in self.steps if isinstance(step, ActionStep) and step.code_action is not None]
        )


class CallbackRegistry:
    """Registry for callbacks that are called at each step of the agent's execution.

    Callbacks are registered by passing a step class and a callback function.
    """

    def __init__(self) -> None:
        self._callbacks: dict[Type[MemoryStep], list[Callable]] = {}

    def register(self, step_cls: Type[MemoryStep], callback: Callable) -> None:
        """Register a callback for a step class.

        Args:
            step_cls (Type[MemoryStep]): Step class to register the callback for.
            callback (Callable): Callback function to register.
        """
        if step_cls not in self._callbacks:
            self._callbacks[step_cls] = []
        self._callbacks[step_cls].append(callback)

    def callback(self, memory_step: MemoryStep, **kwargs: Any) -> None:
        """Call callbacks registered for a step type.

        Args:
            memory_step (MemoryStep): Step to call the callbacks for.
            **kwargs: Additional arguments to pass to callbacks that accept them.
                Typically, includes the agent instance.

        Notes:
            For backwards compatibility, callbacks with a single parameter signature
            receive only the memory_step, while callbacks with multiple parameters
            receive both the memory_step and any additional kwargs.
        """
        # For compatibility with old callbacks that only take the step as an argument
        for cls in memory_step.__class__.__mro__:
            for cb in self._callbacks.get(cls, []):
                cb(memory_step) if len(inspect.signature(cb).parameters) == 1 else cb(memory_step, **kwargs)
