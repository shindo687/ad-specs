"""Exceptions with stable semantics across rule providers."""

from __future__ import annotations

from collections.abc import Callable, Iterable


def _function_name(function: Callable[..., object]) -> str:
    return getattr(function, "__qualname__", repr(function))


class RuleNotFound(LookupError):
    """Raised when no rule is registered for a callable and AD mode."""

    def __init__(self, function: Callable[..., object], mode: str) -> None:
        """Create an error for ``function`` and ``mode`` (JVP or VJP)."""
        self.function = function
        self.mode = mode.upper()
        super().__init__(
            f"No {self.mode} rule is registered for {_function_name(function)}"
        )


class UnsupportedWrt(ValueError):
    """Raised when a rule does not support requested active inputs."""

    def __init__(
        self,
        function: Callable[..., object],
        requested: Iterable[str],
        *,
        supported: Iterable[str] | None = None,
    ) -> None:
        """Describe unsupported ``requested`` names and optional support set."""
        self.function = function
        self.requested = tuple(sorted(requested))
        self.supported = None if supported is None else tuple(sorted(supported))
        message = (
            f"{_function_name(function)} does not support differentiation "
            f"with respect to {self.requested!r}"
        )
        if self.supported is not None:
            message += f"; supported inputs are {self.supported!r}"
        super().__init__(message)


class NonDifferentiablePoint(RuntimeError):
    """Raised when a generally supported rule is invalid at the current point."""
