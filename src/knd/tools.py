import asyncio
import functools
import multiprocessing
import re
import sys
from io import StringIO
from typing import Callable, Set
from urllib.parse import urljoin, urlparse

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from loguru import logger
from pydantic import BaseModel, Field

MAX_DEPTH = 2
MAX_LINKS = 5


async def crawl_url(
    url: str,
    max_depth: int = MAX_DEPTH,
    max_links: int = MAX_LINKS,
    same_domain_only: bool = True,
    prefixes: list[str] | None = None,
    run_config: CrawlerRunConfig | None = None,
    url_filter: Callable[[str], bool] | None = None,
    echo: bool = False,
) -> dict:
    """
    Recursively crawl starting from a URL up to a specified depth.

    Args:
        url: The URL to start crawling from
        max_depth: Maximum depth of recursion (default: 2)
        max_links: Maximum number of links to follow (default: 5)
        same_domain_only: Only follow links within the same domain (default: True)
        prefixes: List of prefixes to follow (default: None). So only links starting with these prefixes will be followed. If same_domain_only is True, it will be automatically added.
        run_config: A CrawlerRunConfig object that specifies the configuration for the crawler.
        url_filter: A function that takes a URL and returns a boolean. If the function returns False, the URL will be skipped.
    """
    visited: Set[str] = set()
    results = {}
    start_domain = urlparse(url).netloc
    prefixes = prefixes or []
    if same_domain_only:
        prefixes = [p for p in prefixes + [start_domain] if urlparse(p).netloc == start_domain]
    if echo:
        logger.info(f"Crawling {url} with prefixes {prefixes}")

    async def _crawl_url(url: str, depth: int):
        if echo:
            logger.info(f"Crawling {url} at depth {depth} with netloc {urlparse(url).netloc}")
        if depth > max_depth or url in visited or len(results) > max_links:
            return

        if depth > 1 and (url_filter and not url_filter(url)):
            if echo:
                logger.warning(f"Skipping {url} because it failed url_filter")
            return

        if prefixes and not any(url.startswith(p) for p in prefixes):
            if echo:
                logger.warning(f"Skipping {url} because it is not in prefixes")
            return

        visited.add(url)

        async with AsyncWebCrawler(verbose=True) as crawler:
            try:
                result = await crawler.arun(url=url, config=run_config)
                results[url] = result.markdown

                # Extract links from the page
                if result.links and depth < max_depth:
                    for _, links in result.links.items():
                        for link in links:
                            if "href" in link:
                                next_url = urljoin(url, link["href"])
                                await _crawl_url(url=next_url, depth=depth + 1)
            except Exception:
                logger.exception(f"Error crawling {url}")

    await _crawl_url(url, 1)
    return results


def crawl_url_sync(
    url: str,
    max_depth: int = MAX_DEPTH,
    max_links: int = MAX_LINKS,
    same_domain_only: bool = True,
    prefixes: list[str] | None = None,
    run_config: CrawlerRunConfig | None = None,
    url_filter: Callable[[str], bool] | None = None,
) -> dict:
    """
    Recursively crawl starting from a URL up to a specified depth.

    Args:
        url: The URL to start crawling from
        max_depth: Maximum depth of recursion (default: 2)
        max_links: Maximum number of links to follow (default: 5)
        same_domain_only: Only follow links within the same domain (default: True)
        prefixes: List of prefixes to follow (default: None). So only links starting with these prefixes will be followed. If same_domain_only is True, it will be automatically added.
        run_config: A CrawlerRunConfig object that specifies the configuration for the crawler.
        url_filter: A function that takes a URL and returns a boolean. If the function returns False, the URL will be skipped.
    """
    return asyncio.run(
        crawl_url(
            url=url,
            max_depth=max_depth,
            max_links=max_links,
            same_domain_only=same_domain_only,
            prefixes=prefixes,
            run_config=run_config,
            url_filter=url_filter,
        )
    )


@functools.lru_cache(maxsize=None)
def warn_once() -> None:
    """Warn once about the dangers of PythonREPL."""
    logger.warning("Python REPL can execute arbitrary code. Use with caution.")


class PythonREPL(BaseModel):
    """Simulates a standalone Python REPL."""

    globals: dict = Field(default_factory=dict, alias="_globals")
    locals: dict = Field(default_factory=dict, alias="_locals")

    @staticmethod
    def sanitize_input(query: str) -> str:
        """Sanitize input to the python REPL.

        Remove whitespace, backtick & python
        (if llm mistakes python console as terminal)

        Args:
            query: The query to sanitize

        Returns:
            str: The sanitized query
        """
        query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
        query = re.sub(r"(\s|`)*$", "", query)
        return query

    @classmethod
    def worker(
        cls,
        command: str,
        queue: multiprocessing.Queue,
        globals: dict | None = None,
        locals: dict | None = None,
    ) -> None:
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        try:
            cleaned_command = cls.sanitize_input(command)
            exec(cleaned_command, globals, locals)
            sys.stdout = old_stdout
            queue.put(mystdout.getvalue())
        except Exception as e:
            sys.stdout = old_stdout
            queue.put(repr(e))

    def run(self, command: str, timeout: int | None = None) -> str:
        """Run command with own globals/locals and returns anything printed.
        Timeout after the specified number of seconds."""

        # Warn against dangers of PythonREPL
        warn_once()

        queue: multiprocessing.Queue = multiprocessing.Queue()

        # Only use multiprocessing if we are enforcing a timeout
        if timeout is not None:
            # create a Process
            p = multiprocessing.Process(target=self.worker, args=(command, queue, self.globals, self.locals))

            # start it
            p.start()

            # wait for the process to finish or kill it after timeout seconds
            p.join(timeout)

            if p.is_alive():
                p.terminate()
                return "Execution timed out"
        else:
            self.worker(command, queue, self.globals, self.locals)
        # get the result from the worker function
        return queue.get()
