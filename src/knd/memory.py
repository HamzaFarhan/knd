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

from knd.ai import MessageCounter, count_messages, trim_messages

AGENT_MEMORIES_DIR = "agent_memories"
MESSAGE_COUNT_LIMIT = 10

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
        # Get superseded IDs
        superseded_ids = set()
        for memory in memories:
            superseded_ids.update(memory.superseded_ids)

        # Create dict to ensure uniqueness by ID
        memory_dict = {}
        for memory in memories:
            if str(memory.id) not in superseded_ids:
                memory_dict[str(memory.id)] = Memory(**memory.model_dump(exclude={"superseded_ids"}))

        # Return sorted list of unique memories
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
Review this interaction and update the agent's accumulated experience, focusing on:

1. Knowledge Evolution:
   - How does this interaction refine our understanding of the domain?
   - What new patterns or anti-patterns emerged?
   - Which existing strategies were reinforced or challenged?

2. Pattern Recognition:
   - Does this case fit known scenarios or represent a new category?
   - What tool usage patterns proved effective or ineffective?
   - What decision points were critical to success/failure?

3. Heuristic Development:
   - What new rules of thumb emerged from this experience?
   - How should existing heuristics be modified based on this case?
   - What contextual factors influenced success?

Integrate this experience with existing knowledge in <agent_experience>:
- Reinforce successful patterns that repeat
- Refine or qualify existing heuristics based on new evidence
- Add new scenarios or edge cases to known patterns
- Update tool usage patterns with new insights
- Identify emerging trends in improvement areas

Focus on building a robust, evolving knowledge base that improves the agent's effectiveness over time.
Remember that this is cumulative experience - don't overwrite existing knowledge, but rather enhance and refine it.
""".strip()


class UserSpecificExperience(BaseModel):
    user_id: UUID = Field(default_factory=uuid4)
    profile: Profile | None
    memories: list[Memory] = Field(default_factory=list)
    summary: str = ""
    message_history: list[_messages.ModelMessage] | None = None

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
    def load_memories(cls, user_dir: Path, new_memories: list[Memory] | None = None) -> list[Memory]:
        user_memories_path = user_dir / "memories.json"
        memories = {
            "memories": [Memory.model_validate(m) for m in json.loads(user_memories_path.read_text())]
            if user_memories_path.exists()
            else []
        }
        memories["memories"].extend(new_memories or [])
        return Memories.model_validate(memories).memories

    @classmethod
    def load_message_history(
        cls,
        user_dir: Path,
        new_messages: list[_messages.ModelMessage] | None = None,
        message_count_limit: int = MESSAGE_COUNT_LIMIT,
    ) -> list[_messages.ModelMessage]:
        message_history_path = user_dir / "message_history.json"
        message_history = (
            _messages.ModelMessagesTypeAdapter.validate_json(message_history_path.read_bytes())
            if message_history_path.exists()
            else []
        )
        return trim_messages(messages=message_history + (new_messages or []), count_limit=message_count_limit)

    @classmethod
    def load(
        cls,
        user_id: UUID,
        agent_dir: Path,
        new_memories: list[Memory] | None = None,
        new_messages: list[_messages.ModelMessage] | None = None,
        message_count_limit: int = MESSAGE_COUNT_LIMIT,
    ) -> Self | None:
        user_dir = agent_dir / f"{user_id}"
        user_profile_path = user_dir / "profile.json"
        user_summary_path = user_dir / "summary.txt"
        if user_profile_path.exists():
            profile = Profile.model_validate_json(user_profile_path.read_text())
        else:
            profile = None
        memories = cls.load_memories(user_dir=user_dir, new_memories=new_memories)
        if user_summary_path.exists():
            summary = user_summary_path.read_text()
        else:
            summary = ""
        message_history = cls.load_message_history(
            user_dir=user_dir, new_messages=new_messages, message_count_limit=message_count_limit
        )
        if not profile and not memories and not summary and not message_history:
            return None
        return cls(profile=profile, memories=memories, summary=summary, message_history=message_history)

    def dump(self, agent_dir: Path, user_id: UUID | None = None) -> None:
        user_dir = agent_dir / f"{user_id or self.user_id}"
        user_dir.mkdir(parents=True, exist_ok=True)
        if self.profile is not None:
            (user_dir / "profile.json").write_text(self.profile.model_dump_json(indent=2))
        if self.memories:
            (user_dir / "memories.json").write_text(
                json.dumps(
                    [m.model_dump() for m in self.load_memories(user_dir=user_dir, new_memories=self.memories)],
                    indent=2,
                )
            )
        if self.summary:
            (user_dir / "summary.txt").write_text(self.summary)
        if self.message_history:
            (user_dir / "message_history.json").write_bytes(
                _messages.ModelMessagesTypeAdapter.dump_json(
                    self.load_message_history(user_dir=user_dir, new_messages=self.message_history), indent=2
                )
            )

    def __str__(self) -> str:
        res = ""
        if self.profile is not None:
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
        user_id: UUID,
        memories_dir: Path | str = AGENT_MEMORIES_DIR,
        message_count_limit: int = MESSAGE_COUNT_LIMIT,
    ) -> Self:
        memories_dir = Path(memories_dir)
        agent_dir = memories_dir / f"{agent_name}"
        agent_experience_path = agent_dir / "agent_experience.json"
        if agent_experience_path.exists():
            agent_experience = AgentExperience.model_validate_json(agent_experience_path.read_text())
        else:
            agent_experience = None
        user_specific_experience = UserSpecificExperience.load(
            user_id=user_id, agent_dir=agent_dir, message_count_limit=message_count_limit
        )
        return cls(
            agent_name=agent_name,
            user_specific_experience=user_specific_experience,
            agent_experience=agent_experience,
        )

    def dump(
        self, memories_dir: Path | str = AGENT_MEMORIES_DIR, agent_name: str = "", user_id: UUID | None = None
    ) -> None:
        memories_dir = Path(memories_dir)
        agent_dir = memories_dir / f"{agent_name or self.agent_name}"
        agent_dir.mkdir(parents=True, exist_ok=True)
        if self.user_specific_experience is not None:
            self.user_specific_experience.dump(agent_dir=agent_dir, user_id=user_id)
        if self.agent_experience is not None:
            (agent_dir / "agent_experience.json").write_text(self.agent_experience.model_dump_json(indent=2))

    @property
    def user_id(self) -> UUID | None:
        if self.user_specific_experience is not None:
            return self.user_specific_experience.user_id
        return None

    @property
    def message_history(self) -> list[_messages.ModelMessage] | None:
        if self.user_specific_experience is not None:
            return self.user_specific_experience.message_history
        return None

    def __str__(self) -> str:
        res = ""
        if self.user_specific_experience is not None and str(self.user_specific_experience):
            res += f"<user_specific_experience>\n{self.user_specific_experience}\n</user_specific_experience>\n\n"
        if self.agent_experience is not None and str(self.agent_experience):
            res += f"<agent_experience>\n{self.agent_experience.model_dump_json()}\n</agent_experience>\n\n"
        return res.strip()


async def summarize(
    memory_agent: Agent,
    message_history: list[_messages.ModelMessage] | None = None,
    summarize_prompt: str = SUMMARY_PROMPT,
    summary_count_limit: int = MESSAGE_COUNT_LIMIT,
    summary_message_counter: MessageCounter = lambda _: 1,
) -> str:
    if not message_history:
        return ""
    if count_messages(messages=message_history, message_counter=summary_message_counter) > summary_count_limit:
        return (
            await memory_agent.run(user_prompt=summarize_prompt, result_type=str, message_history=message_history)
        ).data
    return ""


async def create_user_specific_experience(
    memory_agent: Agent,
    agent_memories: AgentMemories | None,
    message_history: list[_messages.ModelMessage] | None = None,
    new_messages: list[_messages.ModelMessage] | None = None,
    summary_count_limit: int = MESSAGE_COUNT_LIMIT,
    summary_message_counter: MessageCounter = lambda _: 1,
) -> UserSpecificExperience | None:
    if not message_history:
        return None
    log = f"Creating user specific experience for Agent {memory_agent.name}"
    if agent_memories and agent_memories.user_specific_experience:
        log += f" and User {agent_memories.user_specific_experience.user_id}"
    logger.info(log)
    profile_res = await memory_agent.run(
        user_prompt=Profile.user_prompt(), result_type=Profile, message_history=message_history
    )
    profile = profile_res.data
    memories_res = await memory_agent.run(
        user_prompt=Memory.user_prompt(), result_type=list[Memory], message_history=message_history
    )
    memories = memories_res.data
    summary = await summarize(
        memory_agent=memory_agent,
        message_history=message_history,
        summary_count_limit=summary_count_limit,
        summary_message_counter=summary_message_counter,
    )
    user_specific_experience = UserSpecificExperience(
        profile=profile, memories=memories, summary=summary, message_history=new_messages
    )
    return user_specific_experience


async def create_agent_experience(
    memory_agent: Agent, message_history: list[_messages.ModelMessage] | None = None
) -> AgentExperience | None:
    if not message_history:
        return None
    agent_experience_res = await memory_agent.run(
        user_prompt=AgentExperience.user_prompt(), result_type=AgentExperience, message_history=message_history
    )
    return agent_experience_res.data


async def memorize(
    memory_agent: Agent,
    agent_memories: AgentMemories,
    message_history: list[_messages.ModelMessage] | None = None,
    new_messages: list[_messages.ModelMessage] | None = None,
    memories_dir: Path | str = AGENT_MEMORIES_DIR,
    agent_name: str = "",
    user_id: UUID | None = None,
) -> None:
    if not message_history:
        return
    user_specific_experience = await create_user_specific_experience(
        memory_agent=memory_agent,
        agent_memories=agent_memories,
        message_history=message_history,
        new_messages=new_messages,
    )
    agent_experience = await create_agent_experience(memory_agent=memory_agent, message_history=message_history)
    agent_memories.user_specific_experience = user_specific_experience
    agent_memories.agent_experience = agent_experience
    agent_memories.dump(
        memories_dir=memories_dir,
        agent_name=agent_name or agent_memories.agent_name,
        user_id=user_id or agent_memories.user_id,
    )
