from __future__ import annotations as _annotations

from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from pydantic_ai import messages as _messages
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from pydantic_graph.persistence.file import FileStatePersistence


@dataclass
class BillingGraphState:
    billing_agent_message_history: list[_messages.ModelMessage] = field(default_factory=list)
    proof_reader_message_history: list[_messages.ModelMessage] = field(default_factory=list)


@dataclass
class BillingGraphDeps:
    user_id: str


class IsCorrect(BaseModel):
    pass


class CorrectedBillingInfo(BaseModel):
    issues: list[str]
    corrected_billing_info: str

    def __str__(self) -> str:
        return f"<issues>\n{'\n'.join(self.issues)}\n</issues>\n\n<corrected_billing_info>\n{self.corrected_billing_info}\n</corrected_billing_info>"


billing_agent = Agent(
    model="google-gla:gemini-2.0-flash",
    system_prompt=(
        "You are a billing agent. Given a user id, you will check the user's billing information and return the information."
    ),
    deps_type=BillingGraphDeps,
)

proof_reader_agent = Agent(
    model="google-gla:gemini-2.0-flash",
    system_prompt=(
        "You are a proofreader for a billing agent.\n"
        "Given a query and a response, you will check the response for any issues.\n"
        "If there are issues, return them and the corrected billing information in a CorrectedBillingInfo object.\n"
        "If there are no issues, just return an IsCorrect object."
    ),
    deps_type=BillingGraphDeps,
)


@billing_agent.tool_plain
@proof_reader_agent.tool_plain
def get_billed_amount(user_id: str) -> float | str:
    billed_dict = {"1234": 100.0, "5678": 200.0, "9012": 300.0}
    return billed_dict.get(user_id, "User not found")


@billing_agent.system_prompt
@proof_reader_agent.system_prompt
def get_billed_amount_system_prompt(ctx: RunContext[BillingGraphDeps]) -> str:
    return f"<user_id>\n{ctx.deps.user_id}\n</user_id>"


responder_agent = Agent(
    model="google-gla:gemini-2.0-flash",
    system_prompt=(
        "You will face the costumer. Given the original query, the initial response, and the corrected billing information, you will answer the query."
    ),
    deps_type=BillingGraphDeps,
)


@dataclass
class BillingAgent(BaseNode[BillingGraphState, BillingGraphDeps]):
    """Generate a response to the user's query."""

    docstring_notes = True

    query: str

    async def run(self, ctx: GraphRunContext[BillingGraphState, BillingGraphDeps]) -> ProofReader:
        response = await billing_agent.run(
            user_prompt=self.query, message_history=ctx.state.billing_agent_message_history, deps=ctx.deps
        )
        ctx.state.billing_agent_message_history = response.all_messages()
        return ProofReader(query=self.query, response=response.data)


@dataclass
class ProofReader(BaseNode[BillingGraphState, BillingGraphDeps, str]):
    """
    Check and correct the billing information.
    If correct, return the initial response.
    """

    docstring_notes = True

    query: str
    response: str

    async def run(
        self, ctx: GraphRunContext[BillingGraphState, BillingGraphDeps]
    ) -> UpdateBillingAgent | End[str]:
        user_prompt = f"<query>\n{self.query}\n</query>\n\n<response>\n{self.response}\n</response>"
        response = await proof_reader_agent.run(
            user_prompt=user_prompt,
            message_history=ctx.state.proof_reader_message_history,
            deps=ctx.deps,
            result_type=IsCorrect | CorrectedBillingInfo,
        )
        ctx.state.proof_reader_message_history = response.all_messages()
        if isinstance(response.data, IsCorrect):
            return End(self.response)
        else:
            return UpdateBillingAgent(
                query=self.query, response=self.response, corrected_billing_info=response.data
            )


@dataclass
class UpdateBillingAgent(BaseNode[BillingGraphState, BillingGraphDeps, str]):
    """If incorrect, add a feedback message to the billing agent's message history, then return the corrected billing information."""

    docstring_notes = True

    query: str
    response: str
    corrected_billing_info: CorrectedBillingInfo

    async def run(self, ctx: GraphRunContext[BillingGraphState, BillingGraphDeps]) -> End[str]:
        user_prompt = f"<query>\n{self.query}\n</query>\n\n<initial_response>\n{self.response}\n</initial_response>\n\n{str(self.corrected_billing_info)}"
        response = await responder_agent.run(user_prompt=user_prompt, deps=ctx.deps)
        feedback_message = (
            f"The original query was: {self.query}\n\n"
            f"The initial (incorrect) response was: {self.response}\n\n"
            f"The issues with this response are:\n"
            "\n".join(self.corrected_billing_info.issues)
            + "\n\n"
            f"The corrected billing information is: {self.corrected_billing_info.corrected_billing_info}\n\n"
            "Be more careful next time."
        )
        ctx.state.billing_agent_message_history.append(
            _messages.ModelRequest(parts=[_messages.UserPromptPart(content=feedback_message)])
        )
        return End(response.data)


billing_graph = Graph(nodes=[BillingAgent, ProofReader, UpdateBillingAgent], auto_instrument=False)
billing_graph.mermaid_save("billing_graph.jpg", direction="LR", highlighted_nodes=[UpdateBillingAgent])


async def run_graph(user_prompt: str, user_id: str, state_path: Path | None = None) -> str:
    state_path = state_path or Path(f"billing_agent_{str(uuid4())}.json")
    persistence = FileStatePersistence(state_path)
    deps = BillingGraphDeps(user_id=user_id)
    persistence.set_graph_types(billing_graph)
    print((await persistence.load_all())[-1])

    if state_path.exists():
        async with billing_graph.iter_from_persistence(persistence=persistence, deps=deps) as run:
            async for node in run:
                logger.info(node)
        if run.result:
            return run.result.output
        else:
            raise ValueError("No result from graph")

    async with billing_graph.iter(
        start_node=BillingAgent(query=user_prompt), state=BillingGraphState(), deps=deps, persistence=persistence
    ) as run:
        async for node in run:
            logger.info(node)
    if run.result:
        return run.result.output
    else:
        raise ValueError("No result from graph")


if __name__ == "__main__":
    import asyncio

    res = asyncio.run(
        run_graph(
            user_prompt="What is my billed amount?",
            user_id="1234",
            state_path=Path("billing_agent_6a27bc91-b7ed-4245-bcf9-0479dae8af44.json"),
        )
    )
    logger.success(res)
