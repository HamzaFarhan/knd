import asyncio

import logfire
from loguru import logger
from pydantic_ai import Agent

logfire.configure()
logger.configure(handlers=[logfire.loguru_handler()])

agent = Agent(model="google-gla:gemini-1.5-flash", name="knd_evals")


@agent.result_validator
def validate_result(result: str) -> str:
    if "yoo" not in result:
        logfire.error("Result does not contain 'yoo'", _tags=["yoo_check"])
    return result


async def main():
    res = await agent.run(user_prompt="Hello, how are you?")
    logger.info(res.all_messages())


if __name__ == "__main__":
    asyncio.run(main())
