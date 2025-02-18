from datetime import datetime
from uuid import UUID

import pytest
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

from knd.mem_models import Agent, AgentExperience, Memory, Profile, Task, User, validate_memories


def test_profile():
    # Test empty profile
    profile = Profile()
    assert profile.name == ""
    assert profile.age is None
    assert profile.interests == []

    # Test profile with data
    profile = Profile(
        name="Test User",
        age=25,
        interests=["coding", "reading"],
        home="New York",
        occupation="Software Engineer",
        conversation_preferences=["technical topics"],
    )
    assert profile.name == "Test User"
    assert profile.age == 25
    assert profile.interests == ["coding", "reading"]

    # Test string representation
    assert "<user_profile>" in str(profile)
    assert "Test User" in str(profile)

    # Test empty profile string representation
    empty_profile = Profile()
    assert str(empty_profile) == ""


def test_memory():
    # Test memory creation
    memory = Memory(
        context="User mentioned their job",
        category="fact",
        content="Works as a software engineer",
    )
    assert isinstance(memory.id, UUID)
    assert isinstance(memory.created_at, datetime)
    assert memory.context == "User mentioned their job"
    assert memory.category == "fact"
    assert memory.content == "Works as a software engineer"
    assert memory.superseded_ids == []


def test_validate_memories():
    # Create test memories
    memory1 = Memory(
        context="Original job info",
        category="fact",
        content="Works as a teacher",
    )
    memory2 = Memory(
        context="Updated job info",
        category="fact",
        content="Works as a software engineer",
        superseded_ids=[str(memory1.id)],
    )

    # Test validation
    memories = validate_memories([memory1, memory2])
    assert len(memories) == 1
    assert memories[0].content == "Works as a software engineer"

    # Test empty list
    assert validate_memories([]) == []


def test_agent_experience():
    experience = AgentExperience()

    # Test adding to fields
    experience.add_to_field(
        field="effective_strategies",
        value="Break down complex problems into smaller steps",
    )
    assert len(experience.effective_strategies) == 1
    assert "Break down complex" in experience.effective_strategies[0]

    # Test invalid field
    experience.add_to_field(field="invalid_field", value="test")  # Should not raise error


@pytest.mark.asyncio
async def test_user_document():
    # Setup database connection
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    await init_beanie(database=client.test_db, document_models=[User, Agent, Task])
    await User.delete_all()

    # Create and save user
    user = User(profile=Profile(name="Test User"))
    await user.save()

    # Retrieve user
    found_user = await User.find_one({"profile.name": "Test User"})
    assert found_user is not None
    assert found_user.profile.name == "Test User"

    # Update user
    found_user.profile.age = 30
    await found_user.save()

    # Verify update
    updated_user = await User.get(found_user.id)
    assert updated_user.profile.age == 30

    # Cleanup
    await client.drop_database("test_db")


@pytest.mark.asyncio
async def test_agent_document():
    # Setup database connection
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    await init_beanie(database=client.test_db, document_models=[User, Agent, Task])
    await Agent.delete_all()

    # Create and save agent
    agent = Agent(name="Test Agent", model="openai:gpt-4o-mini", description="Test Description")
    await agent.save()

    # Test unique name constraint
    with pytest.raises(Exception):  # Beanie will raise an exception for duplicate name
        duplicate_agent = Agent(name="Test Agent", model="openai:gpt-4o-mini")
        await duplicate_agent.save()

    # Retrieve agent
    found_agent = await Agent.find_one({"name": "Test Agent"})
    assert found_agent is not None
    assert found_agent.description == "Test Description"

    # Cleanup
    await client.drop_database("test_db")


@pytest.mark.asyncio
async def test_task():
    # Setup database connection
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    await init_beanie(database=client.test_db, document_models=[User, Agent, Task])
    await User.delete_all()
    await Agent.delete_all()
    await Task.delete_all()

    # Create necessary objects
    user = User(profile=Profile(name="Test User"))
    await user.save()

    agent = Agent(
        name="Test Agent",
        model="openai:gpt-4o-mini",
        description="Test agent",
    )
    await agent.save()

    # Create task
    task = Task(user=user, agent=agent)
    await task.save()

    # Test message addition
    task.add_message(content="Hello", role="user")
    assert len(task.message_history) == 1
    assert task.message_history[0].parts[0].content == "Hello"
    assert task.message_history[0].parts[0].part_kind == "user-prompt"

    # Test feedback scenario
    task.add_feedback_scenario(
        ai_content="Initial response",
        user_feedback="Please improve this",
        ai_response="Improved response",
    )
    assert len(task.message_history) == 4  # Previous message + 3 new messages
    await task.save()

    # Cleanup
    await client.drop_database("test_db")


@pytest.mark.asyncio
async def test_datetime_validation():
    # Setup database connection
    client = AsyncIOMotorClient("mongodb://localhost:27017")
    await init_beanie(database=client.test_db, document_models=[User, Agent, Task])
    await User.delete_all()
    await Agent.delete_all()
    await Task.delete_all()

    # Test with datetime string
    user = User()
    await user.save()

    agent = Agent(name="Test", model="openai:gpt-4o-mini")
    await agent.save()

    task = Task(
        user=user,
        agent=agent,
        created_at="2024-03-20 12:00:00",
        updated_at="2024-03-20 12:00:00",
    )
    await task.save()
    assert isinstance(task.created_at, datetime)
    assert isinstance(task.updated_at, datetime)

    # Test with datetime object
    now = datetime.now()
    task2 = Task(user=user, agent=agent, created_at=now, updated_at=now)
    await task2.save()
    assert abs(task2.updated_at - now).total_seconds() < 1  # Checks if times are within 1 second

    # Cleanup
    await client.drop_database("test_db")
