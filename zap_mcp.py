import logfire
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerHTTP

logfire.configure()


email_sender = MCPServerHTTP(url="https://actions.zapier.com/mcp/sk-ak-bDOF5iB0wGZnjZWYcx9R9AuSdP/sse")
agent = Agent(model="google-gla:gemini-2.0-flash", mcp_servers=[email_sender])

async with agent.run_mcp_servers():
    result = await agent.run(
        "send an email to fzaidi2014@gmail.com with the subject 'testing zapier mcp' and the body 'this is a test :robot emoji:'"
    )
print(result)
