from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest

import chainrules as ad


def affine(x: float, scale: float = 2.0, *, shift: float = 1.0) -> float:
    return scale * x + shift


@ad.rules.jvp_for(affine)
def affine_jvp(
    tangents: dict[str, object],
    x: float,
    scale: float = 2.0,
    *,
    shift: float = 1.0,
) -> tuple[float, object]:
    value = affine(x, scale, shift=shift)
    total = 0.0
    active = False
    for name, coefficient in (("x", scale), ("scale", x), ("shift", 1.0)):
        direction = tangents.get(name, ad.ZERO)
        if direction is not ad.ZERO:
            total += coefficient * cast(float, direction)
            active = True
    return value, total if active else ad.ZERO


forward_calls = 0
pullback_calls = 0


@ad.rules.vjp_for(affine)
def affine_vjp(
    wrt: tuple[str, ...],
    x: float,
    scale: float = 2.0,
    *,
    shift: float = 1.0,
) -> tuple[float, Callable[[object], dict[str, object]]]:
    global forward_calls
    unsupported = set(wrt) - {"x", "scale", "shift"}
    if unsupported:
        raise ad.UnsupportedWrt(affine, unsupported, supported={"x", "scale", "shift"})
    forward_calls += 1
    value = affine(x, scale, shift=shift)

    def pullback(cotangent: object) -> dict[str, object]:
        global pullback_calls
        pullback_calls += 1
        seed = cast(float, cotangent)
        all_gradients = {"x": seed * scale, "scale": seed * x, "shift": seed}
        return {name: all_gradients[name] for name in wrt}

    return value, pullback


def test_jvp_uses_names_and_preserves_primal_arguments() -> None:
    assert ad.jvp(
        affine,
        3.0,
        shift=4.0,
        tangents={"x": 0.5, "shift": 2.0},
    ) == (10.0, 3.0)


def test_zero_jvp_short_circuits_without_a_registered_rule() -> None:
    def unregistered(x: float) -> float:
        return x**2

    assert ad.jvp(unregistered, 4.0, tangents={}) == (16.0, ad.ZERO)
    assert ad.jvp(unregistered, 4.0, tangents={"x": ad.ZERO}) == (
        16.0,
        ad.ZERO,
    )


def test_nonzero_jvp_requires_a_rule() -> None:
    def unregistered(x: float) -> float:
        return x

    with pytest.raises(ad.RuleNotFound, match="No JVP rule"):
        ad.jvp(unregistered, 1.0, tangents={"x": 1.0})


def test_jvp_validates_mapping_and_parameter_names_before_dispatch() -> None:
    with pytest.raises(TypeError, match="must be a mapping"):
        ad.jvp(affine, 1.0, tangents=cast(object, [1.0]))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="string parameter"):
        ad.jvp(affine, 1.0, tangents=cast(dict[str, object], {1: 1.0}))
    with pytest.raises(TypeError, match="Unknown tangent"):
        ad.jvp(affine, 1.0, tangents={"missing": ad.ZERO})


def bad_jvp_result(x: float) -> float:
    return x


@ad.rules.jvp_for(bad_jvp_result)
def _bad_jvp_result(tangents: object, x: float) -> object:
    del tangents, x
    return "not a pair"


def test_jvp_rejects_invalid_rule_result() -> None:
    with pytest.raises(TypeError, match="two-tuple"):
        ad.jvp(bad_jvp_result, 1.0, tangents={"x": 1.0})


def test_vjp_normalizes_names_preserves_order_and_is_reusable() -> None:
    global forward_calls, pullback_calls
    forward_calls = 0
    pullback_calls = 0
    value, pullback = ad.vjp(affine, 3.0, wrt=("scale", "x"))
    assert value == 7.0
    assert list(pullback(2.0)) == ["scale", "x"]
    assert pullback(3.0) == {"scale": 9.0, "x": 6.0}
    assert forward_calls == 1
    assert pullback_calls == 2


def test_pullback_zero_short_circuits_in_core() -> None:
    global pullback_calls
    pullback_calls = 0
    _, pullback = ad.vjp(affine, 3.0, wrt="x")
    assert pullback(ad.ZERO) == {"x": ad.ZERO}
    assert pullback_calls == 0


@pytest.mark.parametrize(
    ("wrt", "match"),
    [
        ((), "at least one"),
        (("x", "x"), "unique"),
        (("missing",), "Unknown wrt"),
        ((1,), "must be a string"),
    ],
)
def test_vjp_validates_wrt(wrt: object, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        ad.vjp(affine, 1.0, wrt=cast(str | tuple[str, ...], wrt))


def test_vjp_rejects_noniterable_wrt() -> None:
    with pytest.raises(TypeError, match="iterable"):
        ad.vjp(affine, 1.0, wrt=cast(str, 1))


def test_vjp_requires_a_rule() -> None:
    def unregistered(x: float) -> float:
        return x

    with pytest.raises(ad.RuleNotFound, match="No VJP rule"):
        ad.vjp(unregistered, 1.0, wrt="x")


def test_unsupported_wrt_carries_context() -> None:
    error = ad.UnsupportedWrt(affine, {"bad"}, supported={"x"})
    assert error.function is affine
    assert error.requested == ("bad",)
    assert error.supported == ("x",)
    assert "supported inputs" in str(error)


def bad_vjp_pair(x: float) -> float:
    return x


@ad.rules.vjp_for(bad_vjp_pair)
def _bad_vjp_pair(wrt: object, x: float) -> object:
    del wrt, x
    return None


def bad_vjp_pullback(x: float) -> float:
    return x


@ad.rules.vjp_for(bad_vjp_pullback)
def _bad_vjp_pullback(wrt: object, x: float) -> tuple[float, object]:
    del wrt
    return x, 42


def bad_pullback_mapping(x: float) -> float:
    return x


@ad.rules.vjp_for(bad_pullback_mapping)
def _bad_pullback_mapping(
    wrt: object, x: float
) -> tuple[float, Callable[[object], object]]:
    del wrt
    return x, lambda cotangent: cotangent


def wrong_pullback_keys(x: float) -> float:
    return x


@ad.rules.vjp_for(wrong_pullback_keys)
def _wrong_pullback_keys(
    wrt: object, x: float
) -> tuple[float, Callable[[object], dict[str, object]]]:
    del wrt
    return x, lambda cotangent: {"extra": cotangent}


def test_vjp_rejects_invalid_rule_and_pullback_results() -> None:
    with pytest.raises(TypeError, match="two-tuple"):
        ad.vjp(bad_vjp_pair, 1.0, wrt="x")
    with pytest.raises(TypeError, match="callable pullback"):
        ad.vjp(bad_vjp_pullback, 1.0, wrt="x")
    _, nonmapping = ad.vjp(bad_pullback_mapping, 1.0, wrt="x")
    with pytest.raises(TypeError, match="must return a mapping"):
        nonmapping(1.0)
    _, wrong_keys = ad.vjp(wrong_pullback_keys, 1.0, wrt="x")
    with pytest.raises(TypeError, match="missing=.*extra="):
        wrong_keys(1.0)


def test_grad_and_value_and_grad_are_derived_from_one_vjp() -> None:
    global forward_calls
    forward_calls = 0
    assert ad.grad(affine, 3.0, wrt="x") == {"x": 2.0}
    assert forward_calls == 1
    value, gradients = ad.value_and_grad(affine, 3.0, wrt=("x", "scale"))
    assert value == 7.0
    assert gradients == {"x": 2.0, "scale": 3.0}
    assert forward_calls == 2


@pytest.mark.parametrize("result", [True, 1.0 + 2.0j, (1.0, 2.0)])
def test_grad_rejects_non_real_scalar_outputs(result: object) -> None:
    def function(x: float) -> object:
        del x
        return result

    @ad.rules.vjp_for(function)
    def function_vjp(
        wrt: tuple[str, ...], x: float
    ) -> tuple[object, Callable[[object], dict[str, object]]]:
        del x
        return result, lambda cotangent: dict.fromkeys(wrt, cotangent)

    with pytest.raises(TypeError, match="real scalar"):
        ad.grad(function, 1.0, wrt="x")


class OpaqueCallable:
    @property
    def __signature__(self) -> object:
        raise ValueError("opaque")

    def __call__(self, x: float) -> float:
        return x


def test_opaque_callable_requests_a_thin_wrapper() -> None:
    with pytest.raises(TypeError, match="thin Python wrapper"):
        ad.jvp(OpaqueCallable(), 1.0, tangents={})


def test_nondifferentiable_point_preserves_cause() -> None:
    cause = ArithmeticError("singular")
    error = ad.NonDifferentiablePoint("linearization is singular")
    error.__cause__ = cause
    assert error.__cause__ is cause
