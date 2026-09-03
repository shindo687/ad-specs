"""The allocation-free zero tangent sentinel."""

from __future__ import annotations

from typing import Final


class _Zero:
    """Represent a zero tangent or cotangent without allocating storage."""

    __slots__ = ()

    def __repr__(self) -> str:
        """Return the public spelling of the sentinel."""
        return "ZERO"

    def __copy__(self) -> _Zero:
        """Preserve singleton identity when shallow-copied."""
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> _Zero:
        """Preserve singleton identity when deep-copied."""
        del memo
        return self


ZERO: Final = _Zero()
"""The unique allocation-free zero tangent and cotangent value."""
