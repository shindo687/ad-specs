"""Explicit, identity-based registration of JVP and VJP rules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from .errors import RuleNotFound

RuleT = TypeVar("RuleT", bound=Callable[..., object])


@dataclass(frozen=True, slots=True)
class _RuleEntry:
    function: Callable[..., object]
    rule: Callable[..., object]


class RuleRegistry:
    """Map Python callable identities to derivative rules.

    A registry keeps strong references to both functions and rules. Registration
    is intended to happen during import; dispatch is read-only during normal use.
    Separate instances are useful for isolated tests.
    """

    def __init__(self) -> None:
        """Create an empty registry."""
        self._jvp: dict[int, _RuleEntry] = {}
        self._vjp: dict[int, _RuleEntry] = {}

    def jvp_for(self, function: Callable[..., object]) -> Callable[[RuleT], RuleT]:
        """Return a decorator registering one JVP rule for ``function``."""
        return self._decorator(self._jvp, function, "JVP")

    def vjp_for(self, function: Callable[..., object]) -> Callable[[RuleT], RuleT]:
        """Return a decorator registering one VJP rule for ``function``."""
        return self._decorator(self._vjp, function, "VJP")

    def get_jvp(self, function: Callable[..., object]) -> Callable[..., object]:
        """Return the registered JVP rule or raise :class:`RuleNotFound`."""
        return self._get(self._jvp, function, "JVP")

    def get_vjp(self, function: Callable[..., object]) -> Callable[..., object]:
        """Return the registered VJP rule or raise :class:`RuleNotFound`."""
        return self._get(self._vjp, function, "VJP")

    @staticmethod
    def _decorator(
        table: dict[int, _RuleEntry],
        function: Callable[..., object],
        mode: str,
    ) -> Callable[[RuleT], RuleT]:
        key = id(function)

        def decorator(rule: RuleT) -> RuleT:
            if key in table:
                name = getattr(function, "__qualname__", repr(function))
                raise RuntimeError(f"A {mode} rule is already registered for {name}")
            table[key] = _RuleEntry(function, rule)
            return rule

        return decorator

    @staticmethod
    def _get(
        table: dict[int, _RuleEntry],
        function: Callable[..., object],
        mode: str,
    ) -> Callable[..., object]:
        entry = table.get(id(function))
        if entry is None or entry.function is not function:
            raise RuleNotFound(function, mode)
        return entry.rule


rules = RuleRegistry()
"""The process-wide default rule registry."""
