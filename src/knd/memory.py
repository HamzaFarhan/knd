import json
from datetime import datetime
from pathlib import Path
from typing import Self
from uuid import UUID, uuid4

from loguru import logger
from pydantic import UUID4, BaseModel, Field, field_serializer, field_validator
from pydantic.json_schema import SkipJsonSchema
from pydantic_ai import Agent
from pydantic_ai import messages as _messages
from pydantic_ai.result import ResultDataT

from knd.ai import MessageCounter, count_messages

MEMORIES_DIR = "agent_memories"
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


class Profile(BaseModel):
    name: str = ""
    age: int | None = None
    interests: list[str] = Field(default_factory=list)
    home: str = Field(default="", description="Description of the user's home town/neighborhood, etc.")
    occupation: str = Field(default="", description="The user's current occupation or profession")
    conversation_preferences: list[str] = Field(
        default_factory=list,
        description="A list of the user's preferred conversation styles, topics they want to avoid, etc.",
    )

    @classmethod
    def user_prompt(cls) -> str:
        return "Create an updated detailed user profile from the current information you have. Make sure to incorporate the existing profile if it exists in <user_specific_experience>. Prefer to add new stuff to the profile rather than overwrite existing stuff. Unless of course it makes sense to overwrite existing stuff. For example, if the user says they are 25 years old, and the profile says they are 20 years old, then it makes sense to overwrite the profile with the new information."


class Memory(BaseModel):
    "Save notable memories the user has shared with you for later recall."

    id: SkipJsonSchema[UUID4] = Field(default_factory=uuid4)
    created_at: SkipJsonSchema[datetime] = Field(default_factory=datetime.now)
    context: str = Field(
        description="The situation or circumstance where this memory may be relevant. Include any caveats or conditions that contextualize the memory. For example, if a user shares a preference, note if it only applies in certain situations (e.g., 'only at work'). Add any other relevant 'meta' details that help fully understand when and how to use this memory."
    )
    category: str = Field(description="Category of memory (e.g., 'preference', 'fact', 'experience')")
    content: str = Field(description="The specific information, preference, or event being remembered.")
    superseded_ids: list[str] = Field(
        default_factory=list, description="IDs of memories this explicitly supersedes"
    )

    @field_serializer("id")
    def serialize_id(self, id: UUID4) -> str:
        return str(id)

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str | UUID4) -> UUID4:
        if isinstance(v, str):
            return UUID(v)
        return v

    @field_serializer("created_at")
    def serialize_created_at(self, v: datetime) -> str:
        return v.strftime("%Y-%m-%d %H:%M:%S")

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, v: str | datetime) -> datetime:
        if isinstance(v, str):
            return datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
        return v

    @classmethod
    def user_prompt(cls) -> str:
        return """
Analyze the conversation to identify important information that should be remembered for future interactions. Focus on:

1. Personal Details & Preferences:
   - Stated preferences, likes, and dislikes
   - Personal background information
   - Professional or educational details
   - Important relationships mentioned

2. Contextual Information:
   - Time-sensitive information (upcoming events, deadlines)
   - Location-specific details
   - Current projects or goals
   - Recent experiences shared

3. Interaction Patterns:
   - Communication style preferences
   - Topics they enjoy discussing
   - Topics to avoid or handle sensitively
   - Specific terminology or jargon they use

4. Previous Commitments:
   - Promised follow-ups or continuations
   - Unfinished discussions
   - Expressed intentions for future interactions

For each memory identified:
- Include relevant context about when/how it should be used
- Note any temporal limitations or conditions
- Consider how it might affect future interactions

When creating memories that update existing information:
- If you have access to previous memories in <user_specific_experience>, check if any new information contradicts or updates them
- Include the IDs of any superseded memories in the `superseded_ids` field
- Example: If a user previously said they lived in New York (memory ID: abc-123) but now mentions moving to Boston, 
  create a new memory with superseded_ids=["abc-123"]
- Only generate the new memories, the older ones will automatically be overwritten based on the `superseded_ids` field.

Return a list of structured memories, each with clear context, category, and content.
Prioritize information that would be valuable for maintaining conversation continuity across sessions.
""".strip()


class Memories(BaseModel):
    memories: list[Memory] = Field(default_factory=list)

    @field_validator("memories")
    @classmethod
    def validate_memories(cls, memories: list[Memory]) -> list[Memory]:
        if not memories:
            return []
        superseded_ids = set()
        for memory in memories:
            superseded_ids.update(memory.superseded_ids)
        memory_dict = {}
        for memory in memories:
            if str(memory.id) not in superseded_ids:
                memory_dict[str(memory.id)] = Memory(**memory.model_dump(exclude={"superseded_ids"}))
        return sorted(memory_dict.values(), key=lambda x: x.created_at)


class AgentExperience(BaseModel):
    procedural_knowledge: str = Field(
        description="Accumulated understanding of how to approach tasks in the agent's domain"
    )
    common_scenarios: list[str] = Field(
        description="Frequently encountered situations and their typical contexts", default_factory=list
    )
    effective_strategies: list[str] = Field(
        description="Proven approaches and methodologies that have worked well", default_factory=list
    )
    known_pitfalls: list[str] = Field(
        description="Common challenges, edge cases, and how to handle them", default_factory=list
    )
    tool_patterns: list[str] = Field(
        description="Effective ways to use different tools, organized by tool name", default_factory=list
    )
    heuristics: list[str] = Field(
        description="Rules of thumb and decision-making guidelines that emerge from experience",
        default_factory=list,
    )
    user_feedback: list[str] = Field(
        description="Collection of user feedback when the user was not satisfied with the agent's response. This is to help improve the agent's technical skills and behavior. So basic responses from the user are not useful here.",
        default_factory=list,
    )
    improvement_areas: list[str] = Field(
        description="Identified areas for optimization or enhancement. Looking at the user_feedback can also help identify areas for improvement.",
        default_factory=list,
    )

    @classmethod
    def user_prompt(cls) -> str:
        return """
Review this interaction and update the agent's accumulated experience, focusing on general patterns and learnings that apply across all users and sessions:

1. Knowledge Evolution:
   - What general domain insights were gained that could benefit all users?
   - What universal patterns or anti-patterns emerged?
   - Which strategies proved effective regardless of user context?

2. Pattern Recognition:
   - What common scenarios or use cases does this interaction represent?
   - Which tool usage patterns were universally effective?
   - What decision-making principles emerged that could apply broadly?

3. Heuristic Development:
   - What general rules of thumb can be derived from this experience?
   - How can existing heuristics be refined to be more universally applicable?
   - What contextual factors consistently influence success across users?

Integrate this experience with existing knowledge in <agent_experience>:
- Focus on patterns that are likely to repeat across different users
- Develop heuristics that are user-agnostic
- Document tool usage patterns that work in general scenarios
- Identify improvement areas that would benefit all users

Important:
- Exclude user-specific details or preferences
- Focus on technical and procedural knowledge that applies universally
- Capture general principles rather than specific instances
- Maintain privacy by avoiding any personally identifiable information

Focus on building a robust, evolving knowledge base that improves the agent's effectiveness for all users over time.
Remember that this is cumulative experience - don't overwrite existing knowledge, but rather enhance and refine it.
""".strip()


class UserSpecificExperience(BaseModel):
    user_id: UUID = Field(default_factory=uuid4)
    profile: Profile | None = None
    memories: list[Memory] = Field(default_factory=list)
    summary: str = ""
    message_history: list[_messages.ModelMessage] = Field(default_factory=list)

    @field_serializer("user_id")
    def serialize_user_id(self, v: UUID) -> str:
        return str(v)

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str | UUID) -> UUID:
        if isinstance(v, str):
            return UUID(v)
        return v

    @classmethod
    def load(cls, user_id: UUID | str, agent_dir: Path | str, include_profile: bool = True) -> Self | None:
        user_id = UUID(user_id) if isinstance(user_id, str) else user_id
        user_dir = Path(agent_dir) / f"{user_id}"
        if not user_dir.exists():
            return None
        user_profile_path = user_dir / "profile.json"
        user_memories_path = user_dir / "memories.json"
        user_summary_path = user_dir / "summary.txt"
        message_history_path = user_dir / "message_history.json"
        profile = (
            Profile.model_validate_json(user_profile_path.read_text())
            if include_profile and user_profile_path.exists()
            else None
        )
        memories = {
            "memories": [Memory.model_validate(m) for m in json.loads(user_memories_path.read_text())]
            if user_memories_path.exists()
            else []
        }
        memories = Memories.model_validate(memories).memories
        summary = user_summary_path.read_text() if user_summary_path.exists() else ""
        message_history = (
            _messages.ModelMessagesTypeAdapter.validate_json(message_history_path.read_bytes())
            if message_history_path.exists()
            else []
        )
        return cls(
            user_id=user_id, profile=profile, memories=memories, summary=summary, message_history=message_history
        )

    def dump(self, agent_dir: Path | str, user_id: UUID | str | None = None, include_profile: bool = True) -> None:
        user_dir = Path(agent_dir) / f"{user_id or self.user_id or uuid4()}"
        user_dir.mkdir(parents=True, exist_ok=True)
        if include_profile and self.profile:
            (user_dir / "profile.json").write_text(self.profile.model_dump_json(indent=2))
        if self.memories:
            (user_dir / "memories.json").write_text(json.dumps([m.model_dump() for m in self.memories], indent=2))
        if self.summary:
            (user_dir / "summary.txt").write_text(self.summary)
        if self.message_history:
            (user_dir / "message_history.json").write_bytes(
                _messages.ModelMessagesTypeAdapter.dump_json(self.message_history, indent=2)
            )

    def __str__(self) -> str:
        res = ""
        if self.profile:
            res += f"<user_profile>\n{self.profile.model_dump_json()}\n</user_profile>\n\n"
        if self.memories:
            mems = "\n".join([m.model_dump_json(exclude={"superseded_ids"}) for m in self.memories])
            res += f"<memories>\n{mems}\n</memories>\n\n"
        if self.summary:
            res += f"<summary_of_previous_conversations>\n{self.summary}\n</summary_of_previous_conversations>\n\n"
        return res.strip()


class AgentMemories(BaseModel):
    agent_name: str
    user_specific_experience: UserSpecificExperience | None
    agent_experience: AgentExperience | None

    @classmethod
    def load(
        cls,
        agent_name: str,
        user_id: UUID | str | None = None,
        memories_dir: Path | str = MEMORIES_DIR,
        include_profile: bool = True,
    ) -> Self:
        memories_dir = Path(memories_dir or MEMORIES_DIR)
        agent_dir = memories_dir / f"{agent_name}"
        agent_dir.mkdir(parents=True, exist_ok=True)
        agent_experience_path = agent_dir / "agent_experience.json"
        if agent_experience_path.exists():
            agent_experience = AgentExperience.model_validate_json(agent_experience_path.read_text())
        else:
            agent_experience = None
        if not user_id:
            user_specific_experience = None
        else:
            user_specific_experience = UserSpecificExperience.load(
                user_id=UUID(user_id) if isinstance(user_id, str) else user_id,
                agent_dir=agent_dir,
                include_profile=include_profile,
            )
        return cls(
            agent_name=agent_name,
            user_specific_experience=user_specific_experience,
            agent_experience=agent_experience,
        )

    def dump(self, agent_dir: Path | str, user_id: UUID | str | None = None, include_profile: bool = True) -> None:
        agent_dir = Path(agent_dir)
        agent_dir.mkdir(parents=True, exist_ok=True)
        if self.user_specific_experience:
            self.user_specific_experience.dump(
                agent_dir=agent_dir, user_id=user_id, include_profile=include_profile
            )
        if self.agent_experience:
            (agent_dir / "agent_experience.json").write_text(self.agent_experience.model_dump_json(indent=2))

    @property
    def user_id(self) -> UUID | None:
        if self.user_specific_experience:
            return self.user_specific_experience.user_id
        return None

    @property
    def message_history(self) -> list[_messages.ModelMessage]:
        if self.user_specific_experience:
            return self.user_specific_experience.message_history or []
        return []

    def __str__(self) -> str:
        res = ""
        if self.user_specific_experience and str(self.user_specific_experience):
            res += f"<user_specific_experience>\n{self.user_specific_experience}\n</user_specific_experience>\n\n"
        if self.agent_experience and str(self.agent_experience):
            res += f"<agent_experience>\n{self.agent_experience.model_dump_json()}\n</agent_experience>\n\n"
        return res.strip()


async def summarize(
    memory_agent: Agent[None, ResultDataT],
    message_history: list[_messages.ModelMessage] | None = None,
    summarize_prompt: str = SUMMARY_PROMPT,
    summary_count_limit: int = SUMMARY_COUNT_LIMIT,
    summary_message_counter: MessageCounter = lambda _: 1,
) -> str:
    if not message_history:
        return ""
    if count_messages(messages=message_history, message_counter=summary_message_counter) > summary_count_limit:
        return (
            await memory_agent.run(user_prompt=summarize_prompt, result_type=str, message_history=message_history)
        ).data
    else:
        logger.info("Skipping summary because the `message_history` is too short")
    return ""


async def create_user_specific_experience(
    memory_agent: Agent[None, ResultDataT],
    user_id: UUID | str | None = None,
    agent_dir: Path | str = "",
    message_history: list[_messages.ModelMessage] | None = None,
    new_messages: list[_messages.ModelMessage] | None = None,
    summary_count_limit: int = SUMMARY_COUNT_LIMIT,
    summary_message_counter: MessageCounter = lambda _: 1,
    include_profile: bool = True,
) -> UserSpecificExperience | None:
    if not message_history:
        return None
    user_id = user_id or uuid4()
    if isinstance(user_id, str):
        user_id = UUID(user_id)
    logger.info(f"Creating user specific experience for User: {user_id}")

    user_specific_experience = UserSpecificExperience(user_id=user_id)
    if agent_dir:
        user_specific_experience = (
            UserSpecificExperience.load(user_id=user_id, agent_dir=agent_dir, include_profile=include_profile)
            or user_specific_experience
        )

    if include_profile:
        profile = (
            await memory_agent.run(
                user_prompt=Profile.user_prompt(), result_type=Profile, message_history=message_history
            )
        ).data
        user_specific_experience.profile = profile or user_specific_experience.profile
    memories = (
        await memory_agent.run(
            user_prompt=Memory.user_prompt(), result_type=list[Memory], message_history=message_history
        )
    ).data
    user_specific_experience.memories.extend(memories)
    summary = await summarize(
        memory_agent=memory_agent,
        message_history=message_history,
        summary_count_limit=summary_count_limit,
        summary_message_counter=summary_message_counter,
    )
    user_specific_experience.summary = summary
    user_specific_experience.message_history.extend(new_messages or [])
    return user_specific_experience


async def create_agent_experience(
    memory_agent: Agent[None, ResultDataT], message_history: list[_messages.ModelMessage] | None = None
) -> AgentExperience | None:
    if not message_history:
        return None
    agent_experience_res = await memory_agent.run(
        user_prompt=AgentExperience.user_prompt(), result_type=AgentExperience, message_history=message_history
    )
    return agent_experience_res.data


async def memorize(
    memory_agent: Agent[None, ResultDataT],
    agent_memories: AgentMemories,
    message_history: list[_messages.ModelMessage] | None = None,
    new_messages: list[_messages.ModelMessage] | None = None,
    summary_count_limit: int = SUMMARY_COUNT_LIMIT,
    summary_message_counter: MessageCounter = lambda _: 1,
    memories_dir: Path | str = MEMORIES_DIR,
    agent_name: str = "",
    user_id: UUID | str | None = None,
    include_profile: bool = True,
) -> None:
    if not message_history:
        return
    agent_dir = Path(memories_dir or MEMORIES_DIR) / f"{agent_name or agent_memories.agent_name}"
    agent_dir.mkdir(parents=True, exist_ok=True)
    user_specific_experience = await create_user_specific_experience(
        memory_agent=memory_agent,
        user_id=user_id or agent_memories.user_id,
        agent_dir=agent_dir,
        message_history=message_history,
        new_messages=new_messages,
        summary_count_limit=summary_count_limit,
        summary_message_counter=summary_message_counter,
        include_profile=include_profile,
    )
    agent_experience = await create_agent_experience(memory_agent=memory_agent, message_history=message_history)
    agent_memories.user_specific_experience = user_specific_experience
    agent_memories.agent_experience = agent_experience
    agent_memories.dump(agent_dir=agent_dir, include_profile=include_profile)
