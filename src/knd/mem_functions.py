from typing import Any

from loguru import logger
from pydantic_ai import Agent
from pydantic_ai import messages as _messages
from pydantic_ai.result import ResultDataT

from knd.ai import MessageCounter, count_messages
from knd.mem_models import AgentExperience, GeneratedUser, Memory, Profile

MessageHistoryT = list[_messages.ModelMessage | dict[str, Any]] | None

# MEMORIES_DIR = "agent_memories"
SUMMARY_COUNT_LIMIT = 10
SUMMARY_PROMPT = """
Summarize the conversation so far. Make sure to incorporate the existing summary if it exists.
Take a deep breath and think step by step about how to best accomplish this goal using the following steps.
I want the summary to look like this:

<summary>
  Combine the essence of the conversation into a paragraph within this section.
</summary>

<main_points>
  Output the 10-20 most significant points of the conversation in order as a numbered list within this section.
</main_points>

<takeaways>
  Output the 5-10 most impactful takeaways or actions resulting from the conversation within this section.
</takeaways>

<special_instructions_for_tool_calls>
  - Ignore the technical details of tool calls such as arguments or tool names.
  - Focus only on the user's input and the AI's meaningful responses.
  - Retain the exact terms, names, and details from the conversation in the summary.
  - Include any results or outputs from the AI's responses (e.g., "The weather in Paris is sunny with a high of 25°C.").
</special_instructions_for_tool_calls>

- Avoid redundant or repeated items in any output section.
- Focus on the context and key ideas, avoiding unnecessary details or tangents.
""".strip()


def prepare_message_history(message_history: MessageHistoryT) -> list[_messages.ModelMessage]:
    if not message_history:
        return []
    return _messages.ModelMessagesTypeAdapter.validate_python(
        _messages.ModelMessagesTypeAdapter.dump_python(message_history)  # type: ignore
    )


async def summarize(
    memory_agent: Agent[None, ResultDataT],
    message_history: MessageHistoryT = None,
    summarize_prompt: str = SUMMARY_PROMPT,
    summary_count_limit: int = SUMMARY_COUNT_LIMIT,
    summary_message_counter: MessageCounter = lambda _: 1,
) -> str:
    if not message_history:
        return ""
    prepared_message_history = prepare_message_history(message_history)
    if (
        count_messages(messages=prepared_message_history, message_counter=summary_message_counter)
        > summary_count_limit
    ):
        return (
            await memory_agent.run(
                user_prompt=summarize_prompt, result_type=str, message_history=prepared_message_history
            )
        ).data
    else:
        logger.info("Skipping summary because the `message_history` is too short")
    return ""


async def create_user_specific_experience(
    memory_agent: Agent[None, ResultDataT],
    message_history: MessageHistoryT = None,
    include_profile: bool = True,
) -> GeneratedUser:
    if not message_history:
        return GeneratedUser()
    prepared_message_history = prepare_message_history(message_history)
    profile = None
    if include_profile:
        try:
            profile = (
                await memory_agent.run(
                    user_prompt=Profile.user_prompt(),
                    result_type=Profile,
                    message_history=prepared_message_history,
                )
            ).data
        except Exception as e:
            logger.error(f"Error creating profile: {e}")
    try:
        memories = (
            await memory_agent.run(
                user_prompt=Memory.user_prompt(),
                result_type=list[Memory],
                message_history=prepared_message_history,
            )
        ).data
    except Exception as e:
        logger.error(f"Error creating memories: {e}")
        memories = []
    return GeneratedUser(profile=profile, memories=memories)


async def create_agent_experience(
    memory_agent: Agent[None, ResultDataT], message_history: MessageHistoryT = None
) -> AgentExperience | None:
    if not message_history:
        return None
    try:
        agent_experience_res = await memory_agent.run(
            user_prompt=AgentExperience.user_prompt(),
            result_type=AgentExperience,
            message_history=prepare_message_history(message_history),
        )
        return agent_experience_res.data
    except Exception as e:
        logger.error(f"Error creating agent experience: {e}")
        return None
