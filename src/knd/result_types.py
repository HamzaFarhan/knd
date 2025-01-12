from typing import Any

from pydantic import BaseModel, Field


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
""".strip()
