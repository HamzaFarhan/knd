from __future__ import annotations as _annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai import messages as _messages
from pydantic_graph import BaseNode, End, Graph, GraphRunContext
from pydantic_graph.persistence.file import FileStatePersistence
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.theme import Theme

custom_theme = Theme(
    {
        "prompt": "cyan bold",
        "feedback": "bright_yellow",
        "ideas": "bright_green",
        "poem": "bright_magenta",
        "translation": "bright_cyan",
        "title": "white bold",
    }
)
console = Console(theme=custom_theme)


def user_message(content: str) -> _messages.ModelRequest:
    return _messages.ModelRequest(parts=[_messages.UserPromptPart(content=content)])


async def generate_experience(message_history: list[_messages.ModelMessage], experience: str = "") -> str:
    experience = experience or "None so far"
    experience_agent = Agent(
        model="google-gla:gemini-2.0-flash",
        system_prompt=(
            "Given the message history and the experiences, update the experiences.\n"
            "These experiences will be fed back to the agent in the next run.\n"
            "So experience is not actually about what the agent has done or can do.\n"
            "It's more about learning from past runs and user feedback.\n"
            "Like humans, agents can learn from their mistakes and improve.\n"
            "Return only the experience string, no introduction or anything else. Not even the tags.\n"
        ),
    )
    message_history = _messages.ModelMessagesTypeAdapter.dump_python(message_history, mode="json")
    user_prompt = f"<experience>\n{experience}\n</experience>\n<messages>\n{message_history}\n</messages>"
    exp = await experience_agent.run(user_prompt=user_prompt)
    return exp.data


@dataclass
class PoemGraphState:
    ideas_guy_message_history: list[_messages.ModelMessage] = field(default_factory=list)
    poet_message_history: list[_messages.ModelMessage] = field(default_factory=list)
    translator_message_history: list[_messages.ModelMessage] = field(default_factory=list)
    ideas_guy_experience: str = ""
    poet_experience: str = ""
    translator_experience: str = ""


@dataclass
class PoemGraphDeps:
    verify_ideas: bool = True
    verify_poem: bool = True
    verify_translation: bool = True


@dataclass
class Poem:
    en: str
    fr: str


@dataclass
class IdeasGuy(BaseNode[PoemGraphState, PoemGraphDeps]):
    topic: str

    async def run(self, ctx: GraphRunContext[PoemGraphState, PoemGraphDeps]) -> VerifyIdeas | Poet:
        ideas_guy = Agent(
            model="google-gla:gemini-2.0-flash",
            system_prompt=(
                "Given a topic for a poem, write down some ideas for the poem.\n"
                "Your ideas will be used by a poet to write a poem."
            ),
            result_type=list[str],
        )
        user_prompt = f"<topic>\n{self.topic}\n</topic>"
        if ctx.state.ideas_guy_experience:
            user_prompt += f"\n<your_experience>\n{ctx.state.ideas_guy_experience}\n</your_experience>"
        ideas = await ideas_guy.run(user_prompt=user_prompt, message_history=ctx.state.ideas_guy_message_history)
        ctx.state.ideas_guy_message_history = ideas.all_messages()
        if ctx.deps.verify_ideas:
            return VerifyIdeas(topic=self.topic, ideas=ideas.data)
        return Poet(topic=self.topic, ideas=ideas.data)


@dataclass
class VerifyIdeas(BaseNode[PoemGraphState, PoemGraphDeps]):
    topic: str
    ideas: list[str]

    async def run(self, ctx: GraphRunContext[PoemGraphState, PoemGraphDeps]) -> IdeasGuy | Poet:
        ideas_text = "\n".join([f"• {idea}" for idea in self.ideas])
        console.print(
            Panel(
                f"[ideas]{ideas_text}[/]",
                title=f"[title]Ideas for: {self.topic}[/]",
                title_align="left",
                border_style="bright_green",
            )
        )
        feedback = Prompt.ask(
            "[prompt]Happy with these ideas? If yes, just press enter. If no, write your feedback[/]"
        )
        if feedback.strip():
            console.print(Panel(f"[feedback]{feedback}[/]", title="Your Feedback", title_align="left"))
            ctx.state.ideas_guy_message_history.append(user_message(feedback))
            ctx.state.ideas_guy_experience = await generate_experience(
                message_history=ctx.state.ideas_guy_message_history, experience=ctx.state.ideas_guy_experience
            )
            return IdeasGuy(topic=self.topic)
        return Poet(topic=self.topic, ideas=self.ideas)


@dataclass
class Poet(BaseNode[PoemGraphState, PoemGraphDeps]):
    topic: str
    ideas: list[str]

    async def run(self, ctx: GraphRunContext[PoemGraphState, PoemGraphDeps]) -> VerifyPoem | Translator:
        poet = Agent(
            model="google-gla:gemini-2.0-flash",
            system_prompt=(
                "You are a poet. Given a topic and ideas, write a poem.\n"
                "Just the poem, no introduction or anything else."
            ),
        )
        user_prompt = f"<topic>\n{self.topic}\n</topic>\n<ideas>\n{self.ideas}\n</ideas>"
        if ctx.state.poet_experience:
            user_prompt += f"\n<your_experience>\n{ctx.state.poet_experience}\n</your_experience>"
        poem = await poet.run(user_prompt=user_prompt, message_history=ctx.state.poet_message_history)
        ctx.state.poet_message_history = poem.all_messages()
        if ctx.deps.verify_poem:
            return VerifyPoem(topic=self.topic, ideas=self.ideas, poem=poem.data)
        return Translator(poem=poem.data)


@dataclass
class VerifyPoem(BaseNode[PoemGraphState, PoemGraphDeps]):
    topic: str
    ideas: list[str]
    poem: str

    async def run(self, ctx: GraphRunContext[PoemGraphState, PoemGraphDeps]) -> Poet | Translator:
        console.print(
            Panel(
                f"[poem]{self.poem}[/]",
                title=f"[title]Poem about: {self.topic}[/]",
                title_align="left",
                border_style="bright_magenta",
            )
        )
        feedback = Prompt.ask(
            "[prompt]Happy with this poem? If yes, just press enter. If no, write your feedback[/]"
        )
        if feedback.strip():
            console.print(Panel(f"[feedback]{feedback}[/]", title="Your Feedback", title_align="left"))
            ctx.state.poet_message_history.append(user_message(feedback))
            ctx.state.poet_experience = await generate_experience(
                message_history=ctx.state.poet_message_history, experience=ctx.state.poet_experience
            )
            return Poet(topic=self.topic, ideas=self.ideas)
        return Translator(poem=self.poem)


@dataclass
class Translator(BaseNode[PoemGraphState, PoemGraphDeps, Poem]):
    poem: str

    async def run(self, ctx: GraphRunContext[PoemGraphState, PoemGraphDeps]) -> VerifyTranslation | End[Poem]:
        translator = Agent(
            model="google-gla:gemini-2.0-flash",
            system_prompt=(
                "Given a poem, translate it into French.\nJust the poem, no introduction or anything else."
            ),
        )
        user_prompt = f"<poem>\n{self.poem}\n</poem>"
        if ctx.state.translator_experience:
            user_prompt += f"\n<your_experience>\n{ctx.state.translator_experience}\n</your_experience>"
        translated_poem = await translator.run(
            user_prompt=user_prompt, message_history=ctx.state.translator_message_history
        )
        ctx.state.translator_message_history = translated_poem.all_messages()
        if ctx.deps.verify_translation:
            return VerifyTranslation(poem_en=self.poem, poem_fr=translated_poem.data)
        return End(Poem(en=self.poem, fr=translated_poem.data))


@dataclass
class VerifyTranslation(BaseNode[PoemGraphState, PoemGraphDeps, Poem]):
    poem_en: str
    poem_fr: str

    async def run(self, ctx: GraphRunContext[PoemGraphState, PoemGraphDeps]) -> Translator | End[Poem]:
        console.print(
            Panel(
                f"[poem]{self.poem_en}[/]",
                title="[title]Original Poem[/]",
                title_align="left",
                border_style="bright_magenta",
            )
        )
        console.print(
            Panel(
                f"[translation]{self.poem_fr}[/]",
                title="[title]French Translation[/]",
                title_align="left",
                border_style="bright_cyan",
            )
        )
        feedback = Prompt.ask(
            "[prompt]Happy with the translation? If yes, just press enter. If no, write your feedback[/]"
        )
        if feedback.strip():
            console.print(Panel(f"[feedback]{feedback}[/]", title="Your Feedback", title_align="left"))
            ctx.state.translator_message_history.append(user_message(feedback))
            ctx.state.translator_experience = await generate_experience(
                message_history=ctx.state.translator_message_history, experience=ctx.state.translator_experience
            )
            return Translator(poem=self.poem_en)
        return End(Poem(en=self.poem_en, fr=self.poem_fr))


async def run_poem_graph(topic: str, run_id: str = "1234") -> Poem:
    poem_graph = Graph(
        nodes=[IdeasGuy, VerifyIdeas, Poet, VerifyPoem, Translator, VerifyTranslation], auto_instrument=False
    )
    state_path = Path(f"poem_{run_id}.json")
    state_path.unlink(missing_ok=True)
    persistence = FileStatePersistence(state_path)
    state = PoemGraphState()
    deps = PoemGraphDeps()

    async with poem_graph.iter(
        start_node=IdeasGuy(topic=topic), state=state, deps=deps, persistence=persistence
    ) as run:
        # we can run all nodes at once or have custom decisions/code even while running the graph based on the current node/state
        async for node in run:
            print(node, "\n")

    # You can access the experiences after the run
    print(f"Ideas Guy Experience: {state.ideas_guy_experience}")
    print(f"Poet Experience: {state.poet_experience}")
    print(f"Translator Experience: {state.translator_experience}")

    # Check if the graph completed successfully
    if run.result is not None:
        return run.result.output
    # If we got here, something went wrong
    raise ValueError(f"Graph did not complete successfully. Result: {run.result}")


async def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a poem and translate it to French")
    # Use add_argument with keyword arguments as per user preferences
    _ = parser.add_argument("topic", type=str, help="Topic for the poem")
    _ = parser.add_argument("--run-id", type=str, default="1234", help="Unique ID for this run")

    args = parser.parse_args()
    topic: str = args.topic
    run_id: str = args.run_id

    try:
        poem = await run_poem_graph(topic=topic, run_id=run_id)
        print("\nFinal Poem:")
        print(f"\nEnglish:\n{poem.en}")
        print(f"\nFrench:\n{poem.fr}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
