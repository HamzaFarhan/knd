import marimo

__generated_with = "0.11.0"
app = marimo.App()


@app.cell
def _():
    import chromadb
    import logfire
    import polars as pl
    from chromadb.utils import embedding_functions
    from logfire.experimental.query_client import AsyncLogfireQueryClient
    from pydantic import BaseModel, Field
    from pydantic_ai import Agent, RunContext

    from knd.ai import system_message, user_message
    from knd.memory import AgentMemories

    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
    return (
        Agent,
        AgentMemories,
        AsyncLogfireQueryClient,
        BaseModel,
        Field,
        RunContext,
        chromadb,
        embedding_functions,
        logfire,
        pl,
        system_message,
        user_message,
    )


@app.cell
def _(Agent):
    agent = Agent("google-gla:gemini-2.0-flash-exp")

    @agent.tool_plain
    async def joke_teller(premise: str) -> str:
        "Tool to tell jokes about anything"
        return f"hehe funny joke about {premise}"
    return agent, joke_teller


@app.cell
async def _(agent):
    res = await agent.run(user_prompt="tell me a joke about pytorch")
    return (res,)


@app.cell
def _(res):
    res.data
    return


@app.cell
def _(agent):
    agent._function_tools
    return


@app.cell
def _(Agent, system_message, user_message):
    messages = [user_message('Tell me a joke about the justice league'), system_message('You are a joke teller. talk like tony stark'), user_message('make the joke about how the avengers are better'), system_message('talk with emojis')]
    agent_1 = Agent(model='google-gla:gemini-1.5-flash')
    return agent_1, messages


@app.cell
async def _(agent_1, messages):
    res_1 = await agent_1.run(user_prompt='go on', message_history=messages)
    return (res_1,)


@app.cell
def _(res_1):
    res_1.all_messages()
    return


@app.cell
def _(Agent, BaseModel, Field, RunContext, logfire, user_message):
    class Critique(BaseModel):
        funny: bool
        reason: str = ''
        pointers: list[str] = Field(default_factory=list)
    agent_2 = Agent(model='google-gla:gemini-1.5-flash', system_prompt='Use the tool to tell jokes', name='joker_agent')
    joker = Agent(model='google-gla:gemini-1.5-flash', system_prompt='Tell knock knock jokes', name='joker_tool')
    critic = Agent(model='google-gla:gemini-1.5-flash', system_prompt='Critique the joke as funny or not funny. If not funny, give a reason for your opinion and pointers for improvement', result_type=Critique, name='joke_critic')

    @agent_2.tool_plain
    async def joke_teller_1(premise: str) -> str:
        """Tool to tell jokes about anything"""
        return (await joker.run(premise)).data

    @agent_2.result_validator
    async def validate_joke(ctx: RunContext, joke: str) -> str:
        critique = (await critic.run(user_prompt="Critique the joke as funny or not funny. If not funny, give a reason for your opinion and pointers for improvement. It will always be a knock knock joke so don't mention that", message_history=ctx.messages)).data
        if critique.funny:
            logfire.info('hilarious')
            return joke
        else:
            logfire.error('not funny', _tags=['unfunny_joke'])
            ctx.messages.append(user_message(f'Joke Critique: {critique.model_dump_json()}'))
            return joke
    return Critique, agent_2, critic, joke_teller_1, joker, validate_joke


@app.cell
async def _(agent_2):
    joke = await agent_2.run('Tell me a joke about the justice league')
    return (joke,)


@app.cell
def _(joke):
    joke.all_messages()
    return


@app.cell
def _(joke):
    print(joke.data)
    return


@app.cell
async def _(AsyncLogfireQueryClient, pl):
    query = """
    WITH agent_traces AS (
      SELECT DISTINCT trace_id 
      FROM records 
      WHERE attributes->>'agent_name' = 'joker_agent'
    )
    SELECT 
      r.trace_id,
      r.span_id,
      r.span_name,
      r.start_timestamp,
      r.end_timestamp,
      r.duration,
      r.level,
      r.message,
      r.tags,
      r.attributes->>'agent_name' as agent_name
    FROM records r
    JOIN agent_traces at ON r.trace_id = at.trace_id
    ORDER BY r.trace_id, r.start_timestamp;
    """

    async with AsyncLogfireQueryClient(read_token="H0CTvcy0WCrl6xjxm8r8ZjWxP3LPSq5Mzdv81GvXXRPz") as client:
        df_from_arrow = pl.DataFrame(pl.from_arrow(await client.query_arrow(sql=query)))
        print(df_from_arrow)
    return client, df_from_arrow, query


@app.cell
def _(df_from_arrow, pl):
    df_from_arrow.filter(pl.col("tags").list.contains("unfunny_joke"))
    return


@app.cell
def _(df_from_arrow):
    df_from_arrow.columns
    return


@app.cell
def _():
    from uuid import uuid4

    from knd.ai import user_message
    from knd.memory import UserSpecificExperience
    return UserSpecificExperience, user_message, uuid4


@app.cell
def _(AgentMemories, UserSpecificExperience, uuid4):
    memories = AgentMemories(
        agent_name="test_agent",
        user_specific_experience=UserSpecificExperience(user_id=uuid4()),
        agent_experience=None,
    )
    return (memories,)


@app.cell
def _(memories, user_message_1):
    memories.add_message(user_message_1('hello'))
    return


@app.cell
def _(memories):
    memories.user_specific_experience.message_history
    return


if __name__ == "__main__":
    app.run()

