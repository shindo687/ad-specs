"""Dependency-free conformance helpers for derivative rule authors.

Finite differences live here only as test oracles. Core dispatch never falls
back to numerical differentiation.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import fields, is_dataclass, replace
from typing import Any, cast

from ._zero import ZERO
from .api import jvp, vjp


def _items(value: object) -> list[tuple[object, object]] | None:
    if isinstance(value, Mapping):
        return list(value.items())
    if isinstance(value, tuple | list):
        return list(enumerate(value))
    if is_dataclass(value) and not isinstance(value, type):
        return [(field.name, getattr(value, field.name)) for field in fields(value)]
    return None


def real_inner(left: object, right: object) -> float:
    """Return the real inner product for scalars, arrays, or nested structures.

    Array leaves are handled by duck typing: they need conjugation,
    multiplication, summation, and conversion of a scalar result to ``float``.
    """
    if left is ZERO or right is ZERO:
        return 0.0
    left_items = _items(left)
    right_items = _items(right)
    if (left_items is None) != (right_items is None):
        raise TypeError("inner-product operands have incompatible structures")
    if left_items is not None and right_items is not None:
        left_map = dict(left_items)
        right_map = dict(right_items)
        if left_map.keys() != right_map.keys():
            raise TypeError("inner-product operands have different fields or keys")
        return sum(real_inner(left_map[key], right_map[key]) for key in left_map)

    left_leaf: Any = left
    right_leaf: Any = right
    conjugated = left_leaf.conjugate() if hasattr(left_leaf, "conjugate") else left_leaf
    product = conjugated * right_leaf
    if hasattr(product, "sum"):
        product = product.sum()
    real_part = product.real if hasattr(product, "real") else product
    if hasattr(real_part, "item"):
        real_part = real_part.item()
    return float(real_part)


def _map_binary(function: Callable[[Any, Any], Any], left: Any, right: Any) -> Any:
    if left is ZERO or right is ZERO:
        return function(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if left.keys() != right.keys():
            raise TypeError("values have different mapping keys")
        return {key: _map_binary(function, left[key], right[key]) for key in left}
    if isinstance(left, tuple) and isinstance(right, tuple):
        if len(left) != len(right):
            raise TypeError("values have different tuple lengths")
        return tuple(
            _map_binary(function, a, b) for a, b in zip(left, right, strict=False)
        )
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise TypeError("values have different list lengths")
        return [_map_binary(function, a, b) for a, b in zip(left, right, strict=False)]
    if is_dataclass(left) and is_dataclass(right):
        changes = {
            field.name: _map_binary(
                function, getattr(left, field.name), getattr(right, field.name)
            )
            for field in fields(left)
        }
        return replace(cast(Any, left), **changes)
    return function(left, right)


def _add_scaled(primal: object, tangent: object, scale: float) -> object:
    def leaf(value: Any, direction: Any) -> Any:
        if direction is ZERO:
            return value
        return value + scale * direction

    return _map_binary(leaf, primal, tangent)


def _subtract(left: object, right: object) -> object:
    def leaf(a: Any, b: Any) -> Any:
        if a is ZERO and b is ZERO:
            return ZERO
        if a is ZERO:
            return -b
        if b is ZERO:
            return a
        return a - b

    return _map_binary(leaf, left, right)


def _divide(value: object, denominator: float) -> object:
    def leaf(item: Any, _other: Any) -> Any:
        if item is ZERO:
            return ZERO
        return item / denominator

    return _map_binary(leaf, value, value)


def _norm(value: object) -> float:
    return math.sqrt(max(0.0, real_inner(value, value)))


def _assert_close(
    actual: object,
    expected: object,
    *,
    atol: float,
    rtol: float,
    context: str,
) -> None:
    error = _norm(_subtract(actual, expected))
    limit = atol + rtol * max(_norm(actual), _norm(expected))
    if error > limit:
        raise AssertionError(f"{context}: error {error:.6g} exceeds {limit:.6g}")


def _perturbed_call(
    function: Callable[..., object],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    directions: Mapping[str, object],
    scale: float,
) -> object:
    bound = inspect.signature(function).bind(*args, **kwargs)
    bound.apply_defaults()
    for name, direction in directions.items():
        bound.arguments[name] = _add_scaled(bound.arguments[name], direction, scale)
    return function(*bound.args, **bound.kwargs)


def assert_jvp_close(
    function: Callable[..., object],
    /,
    *args: object,
    tangents: Mapping[str, object],
    steps: Sequence[float] = (1e-4, 1e-5, 1e-6),
    atol: float = 1e-7,
    rtol: float = 1e-5,
    **kwargs: object,
) -> None:
    """Assert primal parity and JVP agreement with a central-difference oracle."""
    value, tangent_out = jvp(function, *args, tangents=tangents, **kwargs)
    _assert_close(
        value,
        function(*args, **kwargs),
        atol=atol,
        rtol=rtol,
        context="primal parity",
    )
    failures: list[str] = []
    for step in steps:
        plus = _perturbed_call(function, args, kwargs, tangents, step)
        minus = _perturbed_call(function, args, kwargs, tangents, -step)
        difference = _subtract(plus, minus)
        try:
            _assert_close(
                tangent_out,
                _divide(difference, 2.0 * step),
                atol=atol,
                rtol=rtol,
                context=f"JVP finite difference at h={step:g}",
            )
            return
        except AssertionError as error:
            failures.append(str(error))
    raise AssertionError("; ".join(failures))


def assert_vjp_close(
    function: Callable[..., object],
    /,
    *args: object,
    directions: Mapping[str, object],
    cotangent: object,
    steps: Sequence[float] = (1e-4, 1e-5, 1e-6),
    atol: float = 1e-7,
    rtol: float = 1e-5,
    **kwargs: object,
) -> None:
    """Assert VJP agreement with directional finite differences."""
    value, pullback = vjp(function, *args, wrt=tuple(directions), **kwargs)
    _assert_close(
        value,
        function(*args, **kwargs),
        atol=atol,
        rtol=rtol,
        context="primal parity",
    )
    gradients = pullback(cotangent)
    expected = sum(real_inner(directions[name], gradients[name]) for name in directions)
    failures: list[str] = []
    for step in steps:
        plus = _perturbed_call(function, args, kwargs, directions, step)
        minus = _perturbed_call(function, args, kwargs, directions, -step)
        actual = real_inner(_subtract(plus, minus), cotangent) / (2.0 * step)
        limit = atol + rtol * max(abs(actual), abs(expected))
        if abs(actual - expected) <= limit:
            return
        failures.append(
            f"VJP finite difference at h={step:g}: "
            f"error {abs(actual - expected):.6g} exceeds {limit:.6g}"
        )
    raise AssertionError("; ".join(failures))


def assert_jvp_vjp_duality(
    function: Callable[..., object],
    /,
    *args: object,
    tangents: Mapping[str, object],
    cotangent: object,
    atol: float = 1e-7,
    rtol: float = 1e-6,
    **kwargs: object,
) -> None:
    """Assert ``<Jv, y_bar> = <v, J* y_bar>`` for registered rules."""
    _, tangent_out = jvp(function, *args, tangents=tangents, **kwargs)
    _, pullback = vjp(function, *args, wrt=tuple(tangents), **kwargs)
    gradients = pullback(cotangent)
    left = real_inner(tangent_out, cotangent)
    right = sum(real_inner(tangents[name], gradients[name]) for name in tangents)
    limit = atol + rtol * max(abs(left), abs(right))
    if abs(left - right) > limit:
        raise AssertionError(
            f"JVP/VJP duality: error {abs(left - right):.6g} exceeds {limit:.6g}"
        )
