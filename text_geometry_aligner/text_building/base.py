"""Interface for constructing output text from matched ALTO words."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from ..io_alto import ALTOWord


class TextBuilder(ABC):
    """Convert matched ALTO words into output text."""

    @abstractmethod
    def build(self, words: Sequence[ALTOWord]) -> str | None:
        raise NotImplementedError
