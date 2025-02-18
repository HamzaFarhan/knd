from beanie import PydanticObjectId
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.models import KnownModelName

from knd.ai import MessageContentT, MessageRole
from knd.mem_functions import MessageHistoryT, create_agent_experience, create_user_specific_experience, summarize
from knd.mem_models import Agent, Memory, Profile, Task, TaskStatus, User
from knd.mem_models import Agent as AgentDoc

MEMORY_AGENT_MODEL: KnownModelName = "google-gla:gemini-1.5-flash"

router = APIRouter()


class MessageRequest(BaseModel):
    content: MessageContentT
    role: MessageRole = MessageRole.USER


class FeedbackRequest(BaseModel):
    ai_content: MessageContentT
    user_feedback: MessageContentT
    ai_response: MessageContentT = ""


class ExperienceUpdateRequest(BaseModel):
    field: str
    value: str


class MemoryRequest(BaseModel):
    context: str
    category: str
    content: str
    superseded_ids: list[str] = Field(default_factory=list)


class GenerateUserExperienceRequest(BaseModel):
    agent_name: str
    message_history: MessageHistoryT = Field(default_factory=list)


class GenerateAgentExperienceRequest(BaseModel):
    message_history: MessageHistoryT = Field(default_factory=list)


def construct_memory_agent(model: KnownModelName = MEMORY_AGENT_MODEL) -> PydanticAgent:
    return PydanticAgent(model=model, name="memory_agent")


# User endpoints
@router.post("/users/", response_model=User)
async def create_user(profile: Profile | None = None) -> User:
    user = User(profile=profile or Profile())
    await user.insert()
    return user


@router.get("/users/{user_id}", response_model=User)
async def get_user(user_id: PydanticObjectId) -> User:
    user = await User.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.put("/users/{user_id}", response_model=User)
async def update_user(user_id: PydanticObjectId, profile: Profile) -> User:
    user = await User.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.profile = profile
    await user.save()
    return user


@router.post("/users/{user_id}/memories", response_model=User)
async def add_user_memory(user_id: PydanticObjectId, memory: MemoryRequest) -> User:
    user = await User.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    new_memory = Memory(
        context=memory.context,
        category=memory.category,
        content=memory.content,
        superseded_ids=memory.superseded_ids,
    )
    user.memories.append(new_memory)
    await user.save()
    return user


# Agent endpoints
@router.post("/agents/", response_model=Agent)
async def create_agent(agent: Agent) -> Agent:
    await agent.insert()
    return agent


@router.get("/agents/{name}", response_model=Agent)
async def get_agent(name: str) -> Agent:
    agent = await Agent.find_one(Agent.name == name)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.post("/agents/{name}/experience", response_model=Agent)
async def update_agent_experience(name: str, update: ExperienceUpdateRequest) -> Agent:
    agent = await Agent.find_one(Agent.name == name)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    agent.experience.add_to_field(field=update.field, value=update.value)
    await agent.save()
    return agent


# Task endpoints
@router.post("/tasks/", response_model=Task)
async def create_task(user_id: PydanticObjectId, agent_name: str) -> Task:
    user = await User.get(user_id)
    agent = await Agent.find_one(Agent.name == agent_name)
    if not user or not agent:
        raise HTTPException(status_code=404, detail="User or Agent not found")

    task = Task(user=user, agent=agent)  # type: ignore
    await task.insert()
    return task


@router.get("/tasks/{task_id}", response_model=Task)
async def get_task(task_id: PydanticObjectId) -> Task:
    task = await Task.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@router.post("/tasks/{task_id}/message", response_model=Task)
async def add_task_message(task_id: PydanticObjectId, message: MessageRequest) -> Task:
    task = await Task.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task.add_message(content=message.content, role=message.role)
    await task.save()
    return task


@router.post("/tasks/{task_id}/feedback", response_model=Task)
async def add_task_feedback(task_id: PydanticObjectId, feedback: FeedbackRequest) -> Task:
    task = await Task.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task.add_feedback_scenario(
        ai_content=feedback.ai_content, user_feedback=feedback.user_feedback, ai_response=feedback.ai_response
    )
    await task.save()
    return task


@router.put("/tasks/{task_id}/status", response_model=Task)
async def update_task_status(task_id: PydanticObjectId, status: TaskStatus) -> Task:
    task = await Task.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    task.status = status
    await task.save()
    return task


@router.post("/users/{user_id}/experience/generate", response_model=User)
async def generate_user_experience(user_id: PydanticObjectId, request: GenerateUserExperienceRequest) -> User:
    """
    Generate a user-specific experience using an agent.
    The agent is used to create a profile and extract memories from the provided message_history.
    The generated information is then merged into the user document.
    """
    user = await User.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    memory_agent = construct_memory_agent()
    generated_user = await create_user_specific_experience(
        memory_agent=memory_agent, message_history=request.message_history, include_profile=True
    )
    user.update_from_generated_user(generated_user=generated_user)
    await user.save()
    return user


@router.post("/agents/{name}/experience/generate", response_model=AgentDoc)
async def generate_agent_experience(name: str, request: GenerateAgentExperienceRequest) -> AgentDoc:
    """
    Generate an updated agent experience based on the provided conversation history.
    The new experience is merged into the existing agent document.
    """
    agent_doc = await AgentDoc.find_one(AgentDoc.name == name)
    if not agent_doc:
        raise HTTPException(status_code=404, detail="Agent not found")

    memory_agent = construct_memory_agent()
    agent_experience = await create_agent_experience(
        memory_agent=memory_agent, message_history=request.message_history
    )
    if agent_experience is not None:
        agent_doc.experience = agent_experience
        await agent_doc.save()
    return agent_doc


@router.get("/tasks/{task_id}/summary", response_model=str)
async def generate_task_summary(*, task_id: PydanticObjectId) -> str:
    """
    Generate a summary for the specified task's message history.

    This endpoint retrieves the task, creates a memory agent, and then uses the summarize
    function to generate a summary if the message count exceeds the defined limit.
    """
    task = await Task.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    memory_agent = construct_memory_agent()
    summary_text = await summarize(memory_agent=memory_agent, message_history=task.message_history)
    return summary_text


def create_app() -> FastAPI:
    app = FastAPI(title="Knowledge API")
    app.include_router(router, prefix="/api")
    return app
