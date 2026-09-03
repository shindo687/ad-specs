"""Pythonic user-facing operations derived from JVP and VJP rules."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping
from numbers import Real
from typing import TypeVar, cast

from ._zero import ZERO
from .registry import rules

ValueT = TypeVar("ValueT")
Pullback = Callable[[object], dict[str, object]]


def _signature(function: Callable[..., object]) -> inspect.Signature:
    try:
        return inspect.signature(function)
    except (TypeError, ValueError) as error:
        name = getattr(function, "__qualname__", repr(function))
        raise TypeError(
            f"Cannot inspect the signature of {name}; register a thin Python "
            "wrapper with an explicit signature"
        ) from error


def _bind(
    function: Callable[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> inspect.Signature:
    signature = _signature(function)
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    return signature


def _validate_names(
    signature: inspect.Signature, names: Iterable[str], *, label: str
) -> None:
    unknown = set(names) - set(signature.parameters)
    if unknown:
        raise TypeError(f"Unknown {label} parameter names: {sorted(unknown)!r}")


def _normalize_wrt(wrt: str | Iterable[str]) -> tuple[str, ...]:
    names: tuple[str, ...]
    if isinstance(wrt, str):
        names = (wrt,)
    else:
        try:
            names = tuple(wrt)
        except TypeError as error:
            raise TypeError(
                "wrt must be a parameter name or iterable of names"
            ) from error
    if not names:
        raise ValueError("wrt must contain at least one parameter name")
    if any(not isinstance(name, str) for name in names):
        raise TypeError("every wrt entry must be a string parameter name")
    if len(set(names)) != len(names):
        raise ValueError("wrt must contain unique parameter names")
    return names


def _rule_pair(result: object, *, mode: str) -> tuple[object, object]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise TypeError(f"A {mode} rule must return a two-tuple")
    return result


def jvp(
    function: Callable[..., ValueT],
    /,
    *args: object,
    tangents: Mapping[str, object],
    **kwargs: object,
) -> tuple[ValueT, object]:
    """Evaluate a callable and its Jacobian-vector product.

    Args:
        function: Original Python callable whose semantics define the primal.
        *args: Positional primal arguments passed unchanged to ``function``.
        tangents: Tangent values keyed by original parameter name. Omitted names
            are inactive. :data:`ZERO` denotes an allocation-free zero direction.
        **kwargs: Keyword primal arguments passed unchanged to ``function``.

    Returns:
        ``(value, tangent_out)``.

    Raises:
        TypeError: If tangent names do not belong to the callable signature or a
            rule returns an invalid protocol value.
        RuleNotFound: If a nonzero direction is requested without a JVP rule.
    """
    if not isinstance(tangents, Mapping):
        raise TypeError("tangents must be a mapping from parameter names to values")
    signature = _bind(function, args, kwargs)
    if any(not isinstance(name, str) for name in tangents):
        raise TypeError("every tangent key must be a string parameter name")
    _validate_names(signature, tangents, label="tangent")
    active_tangents = dict(tangents)
    if not active_tangents or all(value is ZERO for value in active_tangents.values()):
        return function(*args, **kwargs), ZERO

    rule = rules.get_jvp(cast(Callable[..., object], function))
    value, tangent_out = _rule_pair(rule(active_tangents, *args, **kwargs), mode="JVP")
    return cast(ValueT, value), tangent_out


def vjp(
    function: Callable[..., ValueT],
    /,
    *args: object,
    wrt: str | Iterable[str],
    **kwargs: object,
) -> tuple[ValueT, Pullback]:
    """Evaluate a callable and return a reusable vector-Jacobian pullback.

    Args:
        function: Original Python callable whose semantics define the primal.
        *args: Positional primal arguments passed unchanged to ``function``.
        wrt: One parameter name or an iterable of unique active parameter names.
        **kwargs: Keyword primal arguments passed unchanged to ``function``.

    Returns:
        ``(value, pullback)``. The pullback maps an output cotangent to a dict
        whose keys exactly match ``wrt``.

    Raises:
        TypeError: If names or a rule result violate the protocol.
        RuleNotFound: If no VJP rule is registered.
        UnsupportedWrt: If the rule does not support a requested input.
    """
    names = _normalize_wrt(wrt)
    signature = _bind(function, args, kwargs)
    _validate_names(signature, names, label="wrt")
    rule = rules.get_vjp(cast(Callable[..., object], function))
    value, candidate = _rule_pair(rule(names, *args, **kwargs), mode="VJP")
    if not callable(candidate):
        raise TypeError("A VJP rule must return a callable pullback")
    raw_pullback = cast(Callable[[object], object], candidate)

    def pullback(cotangent: object) -> dict[str, object]:
        if cotangent is ZERO:
            return dict.fromkeys(names, cast(object, ZERO))
        result = raw_pullback(cotangent)
        if not isinstance(result, Mapping):
            raise TypeError("A pullback must return a mapping keyed by wrt names")
        actual = set(result)
        expected = set(names)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise TypeError(
                "Pullback keys must exactly match wrt; "
                f"missing={missing!r}, extra={extra!r}"
            )
        return {name: result[name] for name in names}

    return cast(ValueT, value), pullback


def _require_real_scalar(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("grad and value_and_grad require a single real scalar output")


def grad(
    function: Callable[..., object],
    /,
    *args: object,
    wrt: str | Iterable[str],
    **kwargs: object,
) -> dict[str, object]:
    """Return gradients of a real scalar callable via its VJP rule."""
    value, pullback = vjp(function, *args, wrt=wrt, **kwargs)
    _require_real_scalar(value)
    return pullback(1.0)


def value_and_grad(
    function: Callable[..., ValueT],
    /,
    *args: object,
    wrt: str | Iterable[str],
    **kwargs: object,
) -> tuple[ValueT, dict[str, object]]:
    """Return a real scalar value and gradients from one VJP forward pass."""
    value, pullback = vjp(function, *args, wrt=wrt, **kwargs)
    _require_real_scalar(value)
    return value, pullback(1.0)
