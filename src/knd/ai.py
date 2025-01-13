from copy import deepcopy
from typing import Any

from pydantic_ai import Agent, RunContext, models
from pydantic_ai import messages as _messages
from pydantic_ai import usage as _usage
from pydantic_ai.result import RunResult
from rich.prompt import Prompt


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


def count_tokens(messages: list[_messages.ModelMessage] | _messages.ModelMessage) -> int:
    if isinstance(messages, _messages.ModelMessage):
        messages = [messages]
    return sum(count_message_tokens(message) for message in messages)


def trim_messages(
    messages: list[_messages.ModelMessage], n: int | None = None, remove_system_prompt: bool = True
) -> list[_messages.ModelMessage]:
    n = n or len(messages)
    result = messages[-n:]
    while result:
        if (
            isinstance(result[0], _messages.ModelRequest)
            and result[0].parts
            and isinstance(result[0].parts[0], (_messages.UserPromptPart, _messages.SystemPromptPart))
        ):
            break
        n += 1
        if n > len(messages):
            return []
        result = messages[-n:]
    if remove_system_prompt:
        if (
            result
            and isinstance(result[0], _messages.ModelRequest)
            and any(isinstance(p, _messages.SystemPromptPart) for p in result[0].parts)
        ):
            new_parts: list[_messages.ModelRequestPart] = [
                p for p in result[0].parts if not isinstance(p, _messages.SystemPromptPart)
            ]
            if new_parts:
                result[0] = _messages.ModelRequest(parts=new_parts, kind=result[0].kind)
            else:
                return trim_messages(messages=messages, n=n + 1, remove_system_prompt=remove_system_prompt)
    elif n < len(messages) and any(isinstance(p, _messages.SystemPromptPart) for p in messages[0].parts):
        return [messages[0]] + result
    return result


async def create_run_context(
    agent: Agent,
    user_prompt: str,
    ctx: RunContext | None = None,
    model: models.KnownModelName | models.Model | None = None,
):
    deps = agent._get_deps(ctx.deps if (ctx is not None and agent._deps_type is type(ctx.deps)) else None)
    model_used = await agent._get_model(model)
    return RunContext(
        deps=deps, model=model_used, usage=ctx.usage if ctx is not None else _usage.Usage(), prompt=user_prompt
    )


def replace_system_parts(
    messages: list[_messages.ModelMessage], new_parts: list[_messages.ModelRequestPart]
) -> list[_messages.ModelMessage]:
    messages = deepcopy(messages)
    if not messages:
        return [_messages.ModelRequest(parts=new_parts)]
    msg = messages[0]
    if isinstance(msg, _messages.ModelRequest):
        msg.parts = new_parts + [part for part in msg.parts if not isinstance(part, _messages.SystemPromptPart)]
    return messages


def remove_last_tool_call(messages: list[_messages.ModelMessage], tool_name: str):
    messages = deepcopy(messages)
    for msg in messages[::-1]:
        if isinstance(msg, _messages.ModelResponse) and any(
            isinstance(part, _messages.ToolCallPart) for part in msg.parts
        ):
            msg.parts = [
                part
                for part in msg.parts
                if not (isinstance(part, _messages.ToolCallPart) and part.tool_name == tool_name)
            ]
            return messages
    return messages


async def get_messages_for_agent_tool(
    agent: Agent[None, Any],
    user_prompt: str,
    ctx: RunContext,
    model: models.KnownModelName | models.Model | None = None,
) -> list[_messages.ModelMessage]:
    messages = replace_system_parts(
        new_parts=await agent._sys_parts(
            run_context=await create_run_context(agent=agent, user_prompt=user_prompt, ctx=ctx, model=model)
        ),
        messages=ctx.messages,
    )
    return messages


async def run_until_completion(
    user_prompt: str, agent: Agent, message_history: list[_messages.ModelMessage] | None = None, deps: Any = None
) -> RunResult:
    while True:
        res = await agent.run(user_prompt=user_prompt, deps=deps, message_history=message_history)
        if isinstance(res.data, str):
            user_prompt = Prompt.ask(res.data)
            message_history = res.all_messages()
        else:
            return res
