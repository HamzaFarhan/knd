from copy import deepcopy
from typing import Any

from pydantic_ai import Agent, RunContext, models
from pydantic_ai import messages as _messages
from pydantic_ai import usage as _usage
from pydantic_ai.result import RunResult
from rich.prompt import Prompt


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


def replace_system_parts(new_parts: list[_messages.ModelRequestPart], messages: list[_messages.ModelMessage]):
    messages = deepcopy(messages)
    for msg in messages:
        if isinstance(msg, _messages.ModelRequest) and any(
            isinstance(part, _messages.SystemPromptPart) for part in msg.parts
        ):
            msg.parts = new_parts + [
                part for part in msg.parts if not isinstance(part, _messages.SystemPromptPart)
            ]
            return messages
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
