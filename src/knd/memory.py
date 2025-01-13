from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import UUID4, BaseModel, Field, field_validator
from pydantic.json_schema import SkipJsonSchema


class Profile(BaseModel):
    name: str
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
    initial_params: dict[str, Any] = Field(
        description="If available, initial parameters for the task. Could have been provided by the user or already defined",
        default_factory=dict,
    )
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
Self-reflect on the entire task:
1. What were the key challenges?
2. Which approaches worked well?
3. Which approaches failed?
4. What novel situations were encountered?
5. What patterns emerged?

Then extract the task specific experience.
Incorporate the previous task specific experiences too if relevant. So for example, maybe a previous task failed when the user asked for xyz, but in the latest task the user asked for abc and it worked. We should still include the xyz scenario so that we can avoid failing in the future when the user asks for xyz.
""".strip()
