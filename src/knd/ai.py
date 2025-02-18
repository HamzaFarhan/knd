from copy import deepcopy
from enum import StrEnum
from typing import Any, Callable, Literal

from pydantic_ai import Agent
from pydantic_ai import messages as _messages
from pydantic_ai.result import ResultDataT, RunResult
from pydantic_ai.tools import AgentDepsT
from rich.prompt import Prompt


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    AI = "ai"


MessageContentT = _messages.ModelMessage | dict[str, Any] | str

MessageCounter = Callable[[_messages.ModelMessage], int]


def system_message(content: str) -> _messages.ModelRequest:
    return _messages.ModelRequest(parts=[_messages.SystemPromptPart(content=content)])


def user_message(content: str) -> _messages.ModelRequest:
    return _messages.ModelRequest(parts=[_messages.UserPromptPart(content=content)])


def ai_message(content: str) -> _messages.ModelResponse:
    return _messages.ModelResponse(parts=[_messages.TextPart(content=content)])


def new_message(content: MessageContentT, role: MessageRole) -> _messages.ModelMessage:
    if isinstance(content, dict):
        return _messages.ModelMessagesTypeAdapter.validate_python([content])[0]
    if isinstance(content, str):
        if role == MessageRole.SYSTEM:
            return system_message(content)
        if role == MessageRole.USER:
            return user_message(content)
        if role == MessageRole.AI:
            return ai_message(content)
    return content


def count_part_tokens(part: _messages.ModelRequestPart | _messages.ModelResponsePart) -> int:
    if isinstance(part, (_messages.UserPromptPart, _messages.SystemPromptPart, _messages.TextPart)):
        content = part.content
    elif isinstance(part, _messages.ToolReturnPart):
        content = part.model_response_str()
    elif isinstance(part, _messages.RetryPromptPart):
        content = part.model_response()
    elif isinstance(part, _messages.ToolCallPart):
        content = part.args_as_json_str()
    return int(len(content.split()) / 0.75)


def count_message_tokens(message: _messages.ModelMessage) -> int:
    return sum(count_part_tokens(part) for part in message.parts)


def replace_system_parts(
    message: _messages.ModelRequest, new_parts: list[_messages.ModelRequestPart] | None = None
) -> _messages.ModelRequest | None:
    new_parts = new_parts or []
    non_system_parts = [p for p in message.parts if not isinstance(p, _messages.SystemPromptPart)]
    final_parts = non_system_parts + new_parts
    if final_parts:
        return _messages.ModelRequest(parts=final_parts)
    return None


def count_messages(
    messages: list[_messages.ModelMessage] | _messages.ModelMessage, message_counter: MessageCounter = lambda _: 1
) -> int:
    if not isinstance(messages, list):
        messages = [messages]
    return sum(message_counter(msg) for msg in messages)


def trim_messages(
    messages: list[_messages.ModelMessage],
    count_limit: int | list[int] | None = None,
    message_counter: MessageCounter | list[MessageCounter] = lambda _: 1,
    remove_system_prompt: bool = False,
    strategy: Literal["last", "first"] = "last",
) -> list[_messages.ModelMessage]:
    if not messages:
        return []
    messages = deepcopy(messages)
    if remove_system_prompt and isinstance(messages[0], _messages.ModelRequest):
        new_message = replace_system_parts(messages[0])
        if new_message is not None:
            messages[0] = new_message
        else:
            messages = messages[1:]
    if count_limit is None:
        n = len(messages)
    else:
        if not isinstance(count_limit, list):
            count_limit = [count_limit]
        if not isinstance(message_counter, list):
            message_counter = [message_counter]
        current_counts = [0] * len(count_limit)
        n = 0
        if strategy == "first":
            messages_to_count = messages
        else:
            messages_to_count = messages[::-1]
        for msg in messages_to_count:
            # Calculate all counts for this message
            msg_counts = [counter(msg) for counter in message_counter]

            # Check if adding this message would exceed any limit
            would_exceed = False
            for i, (count, limit, msg_count) in enumerate(zip(current_counts, count_limit, msg_counts)):
                if count + msg_count > limit:
                    would_exceed = True
                    break

            if would_exceed:
                break

            # Update all counts
            for i, msg_count in enumerate(msg_counts):
                current_counts[i] += msg_count
            n += 1
    if strategy == "first":
        if n == 0 and not remove_system_prompt:
            n = 1
        return messages[:n]
    result = messages[-n:]
    while result and n < len(messages):
        if isinstance(result[0], _messages.ModelRequest) and isinstance(
            result[0].parts[0], (_messages.UserPromptPart, _messages.SystemPromptPart)
        ):
            break
        n += 1
        result = messages[-n:]
    if n < len(messages) and isinstance(messages[0].parts[0], _messages.SystemPromptPart):
        return [messages[0]] + result
    return result


async def run_until_completion(
    user_prompt: str,
    agent: Agent[AgentDepsT, ResultDataT],
    message_history: list[_messages.ModelMessage] | None = None,
    deps: AgentDepsT = None,
) -> RunResult[ResultDataT]:
    while True:
        res = await agent.run(user_prompt=user_prompt, deps=deps, message_history=message_history)
        if isinstance(res.data, str):
            user_prompt = Prompt.ask(res.data)
            message_history = res.all_messages()
        else:
            return res
