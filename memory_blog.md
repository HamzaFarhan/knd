# Building AI Agents with Memory using PydanticAI and MongoDB

## Introduction

Memory is a crucial component for AI agents handling complex user interactions. Without it, agents would feel like a frustrating colleague who forgets everything between conversations. But memory is more than just recalling text—it is the **experience** of the agent. Over time, an agent can learn from interactions, refine its responses, and carry over knowledge to new tasks—like a well-trained employee.

In this blog, I'll walk through how I implemented a structured memory system for AI agents using **PydanticAI** and **MongoDB**, emphasizing the *practical* side of things rather than theoretical memory concepts.

## Why Memory Matters for AI Agents

When building AI-driven applications, short-term and long-term memory enable the agent to:

- Recall past interactions in a conversation.  
- Store and retrieve user preferences.  
- Improve user experience by remembering prior discussions.  
- Maintain procedural knowledge to enhance decision-making.  
- Evolve and transfer its expertise over time, just like a well-trained employee.

## Memory Structure

I structured the memory into the following components:

1. **User Profile**  
   Stores persistent details (e.g. name, age, interests, conversation preferences).

2. **Memories**  
   Individual data points extracted from past interactions, categorized as facts, preferences, or experiences.

3. **Agent Experience**  
   Accumulated knowledge from interactions that inform the agent's response strategies, carried over to new environments.

4. **Task History**  
   Tracks workflow progress, user-agent interactions, and intermediate results (e.g., conversation transcripts).

## MongoDB Document Models

Here's how those concepts map onto our MongoDB documents:

```python
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

    def __str__(self) -> str:
        res = ""
        if any(v for v in self.model_dump().values()):
            res += f"<user_profile>\n{self.model_dump_json()}\n</user_profile>"
        return res.strip()

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

    def __str__(self) -> str:
        return self.model_dump_json(exclude={"superseded_ids"})

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

# ---------- MongoDB Documents ----------

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

class User(Document):
    """User document storing profile and accumulated memories"""

    profile: Profile = Field(default_factory=Profile)
    memories: list[Memory] = Field(default_factory=list)

    class Settings:
        name = "users"
        validate_on_save = True

    def __str__(self) -> str:
        res = str(self.profile)
        if self.memories:
            mems = "\n\n".join([str(m) for m in self.memories])
            res += f"\n\n<memories>\n{mems}\n</memories>\n\n"
        return res.strip()

    @field_validator("memories")
    @classmethod
    def validate_memories(cls, v: list[Memory]) -> list[Memory]:
        return validate_memories(v)

    def update_from_generated_user(self, generated_user: GeneratedUser) -> None:
        self.profile = generated_user.profile or self.profile
        self.memories.extend(generated_user.memories)


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
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Settings:
        name = "tasks"
        indexes = ["user._id", "agent.name", "status", "created_at"]
        validate_on_save = True

    def experience_str(self) -> str:
        return (
            f"<agent_experience>\n{self.agent.experience.model_dump_json()}\n</agent_experience>\n\n"  # type: ignore
            f"<user_specific_experience>\n{self.user}\n</user_specific_experience>\n\n"
        )
```

## Memory Update & Flow: Step by Step

Here's the full flow, *with* the update logic. Each step shows how the user and agent data are updated and how everything integrates into the dynamic system prompt for future tasks.

### 1. **User Initiates a Conversation**
A new `Task` instance is created in MongoDB, linking the user and the agent. We store `message_history` there as the conversation unfolds.

### 2. **Agent Retrieves Previous Experience**
When the agent runs, it loads any existing user details and agent experience. This will be injected into the dynamic system prompt.

```python
from pydantic_ai import Agent, RunContext

task_assistant = Agent(
    name="task_assistant",
    model="google-gla:gemini-1.5-flash",
    system_prompt="You are a task assistant. Use your stored knowledge to help.",
    deps_type=Task
)

@task_assistant.system_prompt(dynamic=True)
def system_prompt(ctx: RunContext[Task]) -> str:
    # Step #2: Retrieve previous user & agent data
    return ctx.deps.experience_str()
```
  
### 3. **Agent Interacts; `message_history` Grows**
User messages and agent responses append to `message_history`. Memory updates are queued but not processed yet.

### 4. **Memory Agent Processes Conversation**
We have a *separate* agent (often called `memory_agent`) whose job is to analyze the conversation and produce updates to the user's **profile** and **memories**, and to the agent's **experience**. This happens after the conversation.

**Key Insight:** The memory agent processes the *full* message history, which includes the original task agent's dynamic system prompt (from `experience_str()`). This means:

- New memories are created in context of previous agent experience
- Profile updates consider existing user data from prior sessions
- Memory superseding decisions understand the full historical context

```python
async def create_user_experience(memory_agent: Agent, message_history: list):
    prepared_messages = prepare_message_history(message_history)
    # The prepared_messages contain BOTH:
    # 1. The dynamic system prompt with previous experience_str()
    # 2. The new conversation messages
    # 4a. Update the profile (with awareness of existing profile data)
    profile = await memory_agent.run(
        user_prompt=Profile.user_prompt(),
        result_type=Profile,
        message_history=prepared_messages
    )
    # 4b. Generate new memory objects
    memories = await memory_agent.run(
        user_prompt=Memory.user_prompt(),
        result_type=list[Memory],
        message_history=prepared_messages
    )
    return GeneratedUser(profile=profile.data, memories=memories.data)

async def create_agent_experience(memory_agent: Agent, message_history: list):
    prepared_messages = prepare_message_history(message_history)
    # 4c. Identify agent-level heuristics or pitfalls
    return await memory_agent.run(
        user_prompt=AgentExperience.user_prompt(),
        result_type=AgentExperience,
        message_history=prepared_messages
    ).data
```

### 5. **Memories are Validated or Superseded**
Notice that each new `Memory` can reference older memory IDs to override. Once we have a final list of new memories, we integrate them, dropping old ones that were flagged as superseded.

### 6. **Save Updated Experience to MongoDB**
Finally, we merge the newly generated user info (profile + memories) and the updated agent experience into the DB.

```python
async def save_experience(
    user: User,
    generated_user: GeneratedUser,
    agent: Agent,
    agent_experience: AgentExperience
):
    user.update_from_generated_user(generated_user)  # overwrites existing profile & updates memories
    agent.experience = agent_experience
    await user.save()
    await agent.save()
```
  
### 7. **Future Tasks See Updated Knowledge**
Any new `Task` for the same user/agent automatically includes the updated memory in the dynamic prompt (step #2). The agent retains or expands upon previous learnings, *even if the user starts a brand-new conversation or context*.

## Example Scenario

Initial Recommendation
   1. User: "Recommend me a movie"
   2. User message added to `Task.message_history`
   3. Agent suggests rom-com based on empty profile
   4. Agent message added to `Task.message_history`

User Feedback During Conversation
   1. User: "I prefer action movies"
   2. User message added to `Task.message_history`
   3. Agent continues conversation without memory/experience updates but *with* an updated `message_history`
   4. Final message: "How about 'Action Blockbuster Title'?"
   5. User approves, task marked complete
   6. Full conversation stored in `Task.message_history`

Post-Conversation Processing
   1. Memory agent analyzes full history:  
      * Updates profile with `interests: ["action movies"]`  
      * Creates memory: "User expressed preference for action movies"  
      * Records agent experience: "Action recommendations get better engagement"

New Task Next Day
   1. System prompt now includes updated preferences  
   2. Agent immediately suggests "Another Action Thriller" based on stored memory  
   3. User gets personalized response without re-explaining preferences

## Conclusion: Building Adaptive Memory Systems

This structured memory approach enables AI agents to evolve through interactions by combining Pydantic's validation with MongoDB's flexibility:

1. **Continuous Learning**  
   Each interaction improves through:  
   - Profile refinements (`User.profile`)  
   - Context-aware memories (`User.memories`)  
   - Accumulated experience (`Agent.experience`)  

2. **Safe State Management**  
   Beanie ODM ensures atomic, validated updates:
   ```python
   await user.save()
   await agent.save()
   ```

3. **Scalable Architecture**  
   MongoDB handles nested data structures and temporal queries efficiently.

The key innovation is our **post-conversation processing flow** that decouples real-time interactions from background memory analysis. This maintains chat responsiveness while enabling complex updates.

**Result**: Users get personalized experiences while developers leverage automatic context injection:
```python
def experience_str(self) -> str:
    return (
        f"<agent_experience>\n{self.agent.experience.model_dump_json()}\n</agent_experience>\n\n"
        f"<user_specific_experience>\n{self.user}\n</user_specific_experience>\n\n"
    )
```