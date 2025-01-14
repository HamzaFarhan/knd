import json
from datetime import datetime
from pathlib import Path
from typing import Self
from uuid import UUID, uuid4

from pydantic import UUID4, BaseModel, Field, field_validator
from pydantic.json_schema import SkipJsonSchema
from pydantic_ai import messages as _messages

from knd.ai import trim_messages

AGENT_MEMORIES_DIR = "agent_memories"
MESSAGE_COUNT_LIMIT = 10


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
- If you have access to previous memories, check if any new information contradicts or updates them
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
        superseded_ids = set()
        for memory in memories:
            superseded_ids.update(memory.superseded_ids)
        return sorted(
            [
                Memory(**m.model_dump(exclude={"superseded_ids"}))
                for m in memories
                if str(m.id) not in superseded_ids
            ],
            key=lambda x: x.created_at,
        )


class TaskSpecificExperience(BaseModel):
    chain_of_thought: str
    initial_situation: str = Field(description="Describe the starting point of the task")
    key_decisions: list[str] = Field(description="List major decision points and reasoning")
    outcomes: list[str] = Field(description="Describe results")
    user_feedback: list[str] = Field(description="User feedback, if any, during the task", default_factory=list)
    lessons_learned: list[str] = Field(description="Extract key learnings")
    success_patterns: list[str] = Field(default_factory=list)
    failure_patterns: list[str] = Field(default_factory=list)
    tool_usage_patterns: list[str] = Field(default_factory=list)
    future_recommendations: list[str] = Field(description="Suggest improvements", default_factory=list)

    @classmethod
    def user_prompt(cls) -> str:
        return """
Self-reflect on the task execution process, focusing on generalizable patterns and procedures:
1. What were the technical/procedural challenges encountered?
2. Which solution approaches or methodologies proved effective?
3. Which technical approaches or patterns failed?
4. What novel technical situations or edge cases were discovered?
5. What reusable patterns or anti-patterns emerged?

Extract task-specific experience while:
- Focusing on procedural knowledge and technical patterns
- Avoiding user-specific details or personal information
- Emphasizing reusable solutions and approaches
- Documenting edge cases and their handling
- Identifying technical constraints and limitations

Consider and incorporate relevant experiences from previous similar tasks:
- Compare solution approaches across different instances
- Document which technical patterns consistently succeed or fail
- Note environmental or contextual factors that influence success
- Identify common pitfalls and their solutions

The goal is to build a knowledge base of procedural patterns and technical solutions that can be applied across different users and situations.
""".strip()


class UserSpecificExperience(BaseModel):
    profile: Profile | None
    memories: list[Memory] = Field(default_factory=list)
    summary: str = ""
    message_history: list[_messages.ModelMessage] = Field(default_factory=list)

    @classmethod
    def load(cls, user_id: UUID, agent_dir: Path, message_count_limit: int = MESSAGE_COUNT_LIMIT) -> Self:
        user_dir = agent_dir / f"{user_id}"
        user_profile_path = user_dir / "profile.json"
        user_memories_path = user_dir / "memories.json"
        user_summary_path = user_dir / "summary.txt"
        user_message_history_path = user_dir / "message_history.json"
        if user_profile_path.exists():
            profile = Profile.model_validate_json(user_profile_path.read_text())
        else:
            profile = None
        if user_memories_path.exists():
            loaded_memories = {
                "memories": [Memory.model_validate_json(m) for m in json.loads(user_memories_path.read_text())]
            }
            memories = Memories.model_validate(loaded_memories).memories
        else:
            memories = []
        if user_summary_path.exists():
            summary = user_summary_path.read_text()
        else:
            summary = ""
        if user_message_history_path.exists():
            message_history = _messages.ModelMessagesTypeAdapter.validate_json(
                user_message_history_path.read_bytes()
            )
            message_history = trim_messages(messages=message_history, count_limit=message_count_limit)
        else:
            message_history = []
        return cls(profile=profile, memories=memories, summary=summary, message_history=message_history)

    def dump(self, user_id: UUID, agent_dir: Path) -> None:
        user_dir = agent_dir / f"{user_id}"
        user_dir.mkdir(parents=True, exist_ok=True)
        if self.profile is not None:
            (user_dir / "profile.json").write_text(self.profile.model_dump_json())
        if self.memories:
            (user_dir / "memories.json").write_text(json.dumps([m.model_dump_json() for m in self.memories]))
        if self.summary:
            (user_dir / "summary.txt").write_text(self.summary)
        if self.message_history:
            (user_dir / "message_history.json").write_bytes(
                _messages.ModelMessagesTypeAdapter.dump_json(self.message_history)
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
    user_specific_experience: UserSpecificExperience | None
    task_specific_experience: TaskSpecificExperience | None

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
        tse_path = agent_dir / "task_specific_experience.json"
        if tse_path.exists():
            tse = TaskSpecificExperience.model_validate_json(tse_path.read_text())
        else:
            tse = None
        user_specific_experience = UserSpecificExperience.load(
            user_id=user_id, agent_dir=agent_dir, message_count_limit=message_count_limit
        )
        return cls(user_specific_experience=user_specific_experience, task_specific_experience=tse)

    def dump(self, agent_name: str, user_id: UUID, memories_dir: Path | str = AGENT_MEMORIES_DIR) -> None:
        memories_dir = Path(memories_dir)
        agent_dir = memories_dir / f"{agent_name}"
        agent_dir.mkdir(parents=True, exist_ok=True)
        if self.user_specific_experience is not None:
            self.user_specific_experience.dump(user_id=user_id, agent_dir=agent_dir)
        if self.task_specific_experience is not None:
            (agent_dir / "task_specific_experience.json").write_text(
                self.task_specific_experience.model_dump_json()
            )

    @property
    def message_history(self) -> list[_messages.ModelMessage]:
        if self.user_specific_experience is not None:
            return self.user_specific_experience.message_history
        return []

    def __str__(self) -> str:
        res = ""
        if self.user_specific_experience is not None and str(self.user_specific_experience):
            res += f"<user_specific_experience>\n{self.user_specific_experience}\n</user_specific_experience>\n\n"
        if self.task_specific_experience is not None and str(self.task_specific_experience):
            res += f"<task_specific_experience>\n{self.task_specific_experience.model_dump_json()}\n</task_specific_experience>\n\n"
        return res.strip()
