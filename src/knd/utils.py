import inspect
import re
import textwrap
from collections.abc import Callable
from functools import wraps
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, TypeVar

T = TypeVar("T")


def flatten(o: Iterable):
    for item in o:
        if isinstance(item, str):
            yield item
            continue
        try:
            yield from flatten(item)
        except TypeError:
            yield item


def to_camel(s: str, sep: str = "_") -> str:
    if sep not in s:
        return s
    return "".join(s.title().split(sep))


def to_snake(s: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


def noop(x=None, *args, **kwargs):  # noqa
    return x


def resolve_data_path(data_path: list[str | Path] | str | Path, file_extension: str | None = None) -> chain:
    if not isinstance(data_path, list):
        data_path = [data_path]
    paths = []
    for dp in flatten(data_path):
        if isinstance(dp, (str, Path)):
            dp = Path(dp)
            if not dp.exists():
                raise Exception(f"Path {dp} does not exist.")
            if dp.is_dir():
                if file_extension:
                    paths.append(dp.glob(f"*.{file_extension}"))
                else:
                    paths.append(dp.iterdir())
            else:
                if file_extension is None or dp.suffix == f".{file_extension}":
                    paths.append([dp])
    return chain(*paths)


def flatten_list(my_list: list) -> list:
    new_list = []
    for x in my_list:
        if isinstance(x, list):
            new_list += flatten_list(x)
        else:
            new_list.append(x)
    return new_list


def deindent(text: str) -> str:
    return textwrap.dedent(inspect.cleandoc(text))


def remove_digits(text: str) -> str:
    return re.sub(r"\d+", "", text)


def update_fn(
    func: Callable[..., T] | None = None,
    /,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]] | Callable[..., T]:
    """Update a function's name and optionally set its docstring.

    Can be used either as a decorator with keyword arguments or as a direct function.

    Args:
        func: The function to update (optional). If provided, updates are applied directly.
             If not provided, returns a decorator.
        name: The new name for the function (optional). If provided, must not be empty.
        description: Optional docstring for the function

    Example:
        # As a function:
        def my_fn(x):
            return x
        updated_fn = update_fn(my_fn, name='hello_there')

        # As a decorator with name:
        @update_fn(name='hello_there')
        def my_fn(x):
            return x

        # As a decorator with name and description:
        @update_fn(name='hello_there', description='Says hello')
        def my_fn(x):
            return x

        # As a decorator with no arguments:
        @update_fn()
        def my_fn(x):
            return x

        # Works with async functions too:
        @update_fn(name='async_hello')
        async def my_async_fn(x):
            return x
    """
    if name is not None and not name:
        raise ValueError("name cannot be empty if provided")

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        if inspect.iscoroutinefunction(fn):

            @wraps(fn)
            async def wrapper(*args: Any, **kwargs: Any) -> T:  # type: ignore
                return await fn(*args, **kwargs)
        else:

            @wraps(fn)
            def wrapper(*args: Any, **kwargs: Any) -> T:
                return fn(*args, **kwargs)

        if name is not None:
            wrapper.__name__ = name
        if description is not None:
            wrapper.__doc__ = description
        return wrapper  # type: ignore

    # If func is provided, apply the decorator directly
    if func is not None:
        return decorator(func)

    # Otherwise return the decorator for use with @ syntax
    return decorator
