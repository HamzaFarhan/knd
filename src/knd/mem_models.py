from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any
from uuid import UUID, uuid4

from beanie import Document, Indexed, Link
from pydantic import UUID4, BaseModel, Field, field_serializer, field_validator
from pydantic.json_schema import SkipJsonSchema
from pydantic_ai import messages as _messages
from pydantic_ai.models import KnownModelName


def validate_datetime(v: str | datetime) -> datetime:
    if isinstance(v, str):
        return datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
    return v


class TaskStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


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

    @field_validator("id")
    @classmethod
    def validate_id(cls, v: str | UUID) -> UUID:
        if isinstance(v, str):
            return UUID(v)
        return v

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, v: str | datetime) -> datetime:
        return validate_datetime(v)

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


def validate_memories(memories: list[Memory]) -> list[Memory]:
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
        default="", description="Accumulated understanding of how to approach tasks in the agent's domain"
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


class User(Document):
    """User document storing profile and accumulated memories"""

    user_id: Annotated[UUID | str, Indexed(unique=True)] = Field(default_factory=uuid4)
    profile: Profile = Field(default_factory=Profile)
    memories: list[Memory] = Field(default_factory=list)

    class Settings:
        name = "users"
        validate_on_save = True

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str | UUID) -> UUID | str:
        if isinstance(v, str):
            if "test" in v.lower():
                return "agent_tester"
            return UUID(v)
        return v

    @field_validator("memories")
    @classmethod
    def validate_memories(cls, v: list[Memory]) -> list[Memory]:
        return validate_memories(v)

    def __str__(self) -> str:
        res = ""
        if self.profile:
            res += f"<user_profile>\n{self.profile.model_dump_json()}\n</user_profile>\n\n"
        if self.memories:
            mems = "\n".join([m.model_dump_json(exclude={"superseded_ids"}) for m in self.memories])
            res += f"<memories>\n{mems}\n</memories>\n\n"
        return res.strip()


class Agent(Document):
    """Agent document storing configuration and accumulated experience"""

    name: Annotated[str, Indexed(unique=True)]
    model: KnownModelName
    description: str = ""
    system_prompt: str = ""
    experience: AgentExperience = Field(default_factory=AgentExperience)

    class Settings:
        name = "agents"
        validate_on_save = True


class Task(Document):
    """Task document tracking workflow progress"""

    user: Link[User]
    agent: Link[Agent]
    status: TaskStatus = TaskStatus.CREATED
    message_history: list[_messages.ModelMessage | dict[str, Any]] = Field(default_factory=list)
    intermediate_results: dict[str, Any] = Field(default_factory=dict)
    result: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Settings:
        name = "tasks"
        indexes = ["user.user_id", "agent.name", "status", "created_at"]
        validate_on_save = True

    @field_validator("message_history")
    @classmethod
    def validate_message_history(cls, v: list[_messages.ModelMessage | dict[str, Any]]) -> list[dict[str, Any]]:
        return _messages.ModelMessagesTypeAdapter.dump_python(v)  # type: ignore

    @field_serializer("message_history", when_used="always")
    def serialize_message_history(self, v: list[dict[str, Any]]) -> list[_messages.ModelMessage]:
        return _messages.ModelMessagesTypeAdapter.validate_python(v)

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, v: str | datetime) -> datetime:
        return validate_datetime(v)

    @field_validator("updated_at")
    @classmethod
    def validate_updated_at(cls, v: str | datetime) -> datetime:
        return validate_datetime(v)


# async def init_db():
#     """Initialize database connection"""
#     client = AsyncIOMotorClient("mongodb://localhost:27017")
#     await init_beanie(database=client.agent_db, document_models=[User, Agent, Task])


# async def create_task(user_id: UUID, agent_name: str) -> Task:
#     """Create a new task"""
#     user = await User.find_one(User.user_id == user_id)
#     agent = await Agent.find_one(Agent.name == agent_name)

#     task = Task(user=user, agent=agent, status=TaskStatus.CREATED)
#     await task.insert()
#     return task


# async def resume_task(task_id: PydanticObjectId) -> Task:
#     """Resume a paused task"""
#     task = await Task.get(task_id)
#     if task.status != TaskStatus.PAUSED:
#         raise ValueError("Task is not paused")

#     # Resume processing using task.state
#     task.status = TaskStatus.RUNNING
#     await task.save()
#     return task


# async def update_agent_experience(agent: Agent, new_experience: dict):
#     """Update agent's accumulated experience"""
#     # Merge new experience with existing
#     agent.experience.common_scenarios.extend(new_experience.get("scenarios", []))
#     agent.experience.user_feedback.extend(new_experience.get("feedback", []))
#     # ... update other fields
#     await agent.save()
