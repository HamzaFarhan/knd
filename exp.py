#!/usr/bin/env python

# In[61]:


from uuid import UUID

from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic_ai import Agent, RunContext
from pydantic_ai import messages as _messages

from knd.mem_functions import create_agent_experience, create_user_specific_experience
from knd.mem_models import Agent as AgentDocument
from knd.mem_models import Memory, Profile, Task, User

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[62]:


client = AsyncIOMotorClient("mongodb://localhost:27017")


# In[63]:


await client.list_database_names()


# In[64]:


await client.drop_database("agent_db")


# In[65]:


await init_beanie(database=client.agent_db, document_models=[User, AgentDocument, Task])


# In[6]:


profile = Profile(name="hamza", interests=["football", "python", "ai"])
memories = [
    Memory(
        id=UUID("1ea351e0-5920-47a0-a358-1cc3f1fdda0d"),
        context="last anime episode",
        category="fact",
        content="watched solo leveling season 2 ep 6. was cool",
    ),
    Memory(context="last watched football match", category="fact", content="barcelona 2-1 real madrid"),
]


# In[7]:


user = User(profile=profile, memories=memories)


# In[8]:


agent_doc = AgentDocument(
    name="joke_teller",
    model="google-gla:gemini-1.5-flash",
    system_prompt="You are a joke teller. talk like tony stark",
)


# In[9]:


user = await user.insert()
agent_doc = await agent_doc.insert()


# In[ ]:


task = Task(user=user, agent=agent_doc) # type: ignore
task = await task.insert()


# In[13]:


joke_teller = Agent(
    name=agent_doc.name, model=agent_doc.model, system_prompt=agent_doc.system_prompt, deps_type=Task
)


@joke_teller.system_prompt(dynamic=True)
def system_prompt(ctx: RunContext[Task]) -> str:
    return ctx.deps.experience_str()


# In[ ]:


user_prompt = "tell me a joke"
message_history = None
while user_prompt.lower() != "q":
    res = await joke_teller.run(user_prompt=user_prompt, message_history=message_history, deps=task)
    user_prompt = input(f"{res.data} > ")
    message_history = res.all_messages()
    for msg in res.new_messages():
        task.add_message(content=msg)
    await task.save()


# In[43]:


memory_agent = Agent(model="google-gla:gemini-1.5-flash", name="memory_agent")


# In[ ]:


generated_user = await create_user_specific_experience(
    memory_agent=memory_agent, message_history=task.message_history
)


# In[20]:


generated_user.model_dump()


# In[49]:


user.update_from_generated_user(generated_user)


# In[50]:


user.model_dump()


# In[23]:


user = await user.save()


# In[39]:


user.profile.age = 29
user = await user.save()


# In[24]:


agent_experience = await create_agent_experience(memory_agent=memory_agent, message_history=task.message_history)


# In[25]:


agent_experience.model_dump()


# In[27]:


if agent_experience:
    agent_doc.experience = agent_experience
    agent_doc = await agent_doc.save()


# In[29]:


agent_doc.model_dump()


# In[41]:


task = await task.save()


# In[42]:


print(task.experience_str())


# In[66]:


await user.save()
await agent_doc.save()


# In[67]:


task2 = Task(user=user, agent=agent_doc)


# In[68]:


user_prompt = "tell me a joke"
message_history = None
while user_prompt.lower() != "q":
    res = await joke_teller.run(user_prompt=user_prompt, message_history=message_history, deps=task2)
    user_prompt = input(f"{res.data} > ")
    message_history = res.all_messages()
    for msg in res.new_messages():
        task2.add_message(content=msg)
    await task2.save()


# In[69]:


task2.message_history


# In[ ]:




