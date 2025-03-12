from datetime import datetime
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai import messages as _messages
from pydantic_graph import End

agent = Agent(model="google-gla:gemini-2.0-flash")


@agent.tool_plain
def get_current_date() -> str:
    "Get today's date"
    return datetime.now().strftime("%d-%m-%Y")


date_formatter = Agent(
    model="google-gla:gemini-2.0-flash",
    system_prompt=(
        "Given a date in 'dd-mm-yyyy' format, reformat it to plain english.\n"
        "For example: 01-01-2025 -> January first, two thousand and twenty five"
    ),
)


async def main() -> None:
    async with agent.iter(user_prompt="is tomorrow valentine's day?") as agent_run:
        node = agent_run.next_node
        all_nodes = [node]
        node_index = 1

        while not isinstance(node, End):
            node = await agent_run.next(node)
            if agent.is_model_request_node(node) and agent.is_call_tools_node(all_nodes[node_index - 1]):
                for part in node.request.parts:
                    if isinstance(part, _messages.ToolReturnPart) and part.tool_name == "get_current_date":
                        part.content = (await date_formatter.run(user_prompt=f"Date: {part.content}")).data
            all_nodes.append(node)
            node_index += 1

        print(agent_run.result.data if agent_run.result else "No result")

        Path("valentines.json").write_bytes(
            _messages.ModelMessagesTypeAdapter.dump_json(agent_run.ctx.state.message_history, indent=2)
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

"No, tomorrow is March 13, 2025, so it is not Valentine's Day. Valentine's Day is on February 14th."

"""
[
  {
    "parts": [
      {
        "content": "is tomorrow valentine's day?",
        "timestamp": "2025-03-11T19:18:34.124084Z",
        "part_kind": "user-prompt"
      }
    ],
    "kind": "request"
  },
  {
    "parts": [
      {
        "tool_name": "get_current_date",
        "args": {},
        "tool_call_id": null,
        "part_kind": "tool-call"
      }
    ],
    "model_name": "gemini-2.0-flash",
    "timestamp": "2025-03-11T19:18:36.772475Z",
    "kind": "response"
  },
  {
    "parts": [
      {
        "tool_name": "get_current_date",
        "content": "March twelfth, two thousand and twenty five\n",
        "tool_call_id": null,
        "timestamp": "2025-03-11T19:18:36.772958Z",
        "part_kind": "tool-return"
      }
    ],
    "kind": "request"
  },
  {
    "parts": [
      {
        "content": "No, tomorrow is March 13, 2025, and Valentine's Day is on February 14.\n",
        "part_kind": "text"
      }
    ],
    "model_name": "gemini-2.0-flash",
    "timestamp": "2025-03-11T19:18:39.792337Z",
    "kind": "response"
  }
]
"""
