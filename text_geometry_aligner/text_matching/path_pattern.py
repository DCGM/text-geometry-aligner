"""JSON path patterns naming the values an option applies to."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import JSONPath

_COMPONENT = re.compile(r"\.?([A-Za-z_][A-Za-z0-9_]*)|\[(\*|[A-Za-z_][A-Za-z0-9_]*|\d+)\]")


@dataclass(frozen=True)
class JSONPathPattern:
    """A JSON path pattern such as ``agents[*].description[*]``.

    List indices are written in brackets: ``[*]`` matches any index, ``[0]``
    that index only, and a name such as ``[i]`` any index, bound to the name
    so that two matches can be told to share it. A leading ``$`` is allowed.
    """

    text: str
    components: tuple[str | int, ...]

    @classmethod
    def parse(cls, text: str) -> "JSONPathPattern":
        body = text[1:] if text.startswith("$") else text
        components: list[str | int] = []
        position = 0
        while position < len(body):
            match = _COMPONENT.match(body, position)
            if match is None or (match.group(1) is not None and components and not body.startswith(".", position)):
                raise ValueError(f"Invalid JSON path pattern {text!r} at {position}")
            key, index = match.groups()
            if key is not None:
                components.append(key)
            elif index.isdigit():
                components.append(int(index))
            else:
                # An index wildcard or a bound name, told apart from a key
                # by being written in brackets.
                components.append(f"[{index}]")
            position = match.end()
        if not components:
            raise ValueError("JSON path pattern must not be empty")
        return cls(text=text, components=tuple(components))

    def match(self, path: JSONPath, *, prefix: bool = False) -> tuple[tuple[str, int], ...] | None:
        """The indices bound by name when `path` matches, else None.

        With `prefix`, the pattern matches every path under it as well.
        """

        if len(path) < len(self.components) if prefix else len(path) != len(self.components):
            return None
        bound: dict[str, int] = {}
        for component, part in zip(self.components, path):
            if isinstance(component, int) or not component.startswith("["):
                if component != part:
                    return None
            elif not isinstance(part, int):
                return None
            elif component != "[*]":
                name = component[1:-1]
                if bound.setdefault(name, part) != part:
                    return None
        return tuple(sorted(bound.items()))

    def overlaps(self, other: "JSONPathPattern") -> bool:
        """Whether some path of the same length matches both patterns."""

        if len(self.components) != len(other.components):
            return False
        return all(_compatible(first, second) for first, second in zip(self.components, other.components))


def _compatible(first: str | int, second: str | int) -> bool:
    first_index = isinstance(first, int) or first.startswith("[")
    second_index = isinstance(second, int) or second.startswith("[")
    if first_index != second_index:
        return False
    if isinstance(first, int) and isinstance(second, int):
        return first == second
    return first_index or first == second
