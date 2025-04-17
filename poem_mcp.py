from mcp.server.fastmcp import FastMCP
from pydantic_ai import Agent

server = FastMCP("PydanticAI Server")
server_agent = Agent(model="google-gla:gemini-2.0-flash", system_prompt="always reply in rhyme")


@server.tool()
async def poet(theme: str) -> str:
    """Poem generator"""
    r = await server_agent.run(f"write a poem about {theme}")
    return r.data


if __name__ == "__main__":
    server.run()
