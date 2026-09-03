from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest

import chainrules as ad
from chainrules.testing import (
    assert_jvp_close,
    assert_jvp_vjp_duality,
    assert_vjp_close,
    real_inner,
)


def quadratic(x: float, scale: float = 1.0) -> float:
    return scale * x**2


@ad.rules.jvp_for(quadratic)
def quadratic_jvp(
    tangents: dict[str, object], x: float, scale: float = 1.0
) -> tuple[float, float]:
    dx = cast(float, tangents.get("x", 0.0))
    dscale = cast(float, tangents.get("scale", 0.0))
    return quadratic(x, scale), 2.0 * scale * x * dx + x**2 * dscale


@ad.rules.vjp_for(quadratic)
def quadratic_vjp(
    wrt: tuple[str, ...], x: float, scale: float = 1.0
) -> tuple[float, object]:
    def pullback(cotangent: object) -> dict[str, object]:
        seed = cast(float, cotangent)
        values = {"x": seed * 2.0 * scale * x, "scale": seed * x**2}
        return {name: values[name] for name in wrt}

    return quadratic(x, scale), pullback


def test_conformance_helpers_accept_a_correct_rule_pair() -> None:
    assert_jvp_close(quadratic, 3.0, tangents={"x": 0.25})
    assert_vjp_close(
        quadratic,
        3.0,
        directions={"x": 0.25},
        cotangent=2.0,
    )
    assert_jvp_vjp_duality(
        quadratic,
        3.0,
        tangents={"x": 0.25},
        cotangent=2.0,
    )


def test_real_inner_supports_complex_and_nested_structures() -> None:
    assert real_inner(1.0 + 2.0j, 3.0 + 4.0j) == 11.0
    assert (
        real_inner(
            {"a": (1.0, 2.0), "b": ad.ZERO},
            {"a": (3.0, 4.0), "b": 9.0},
        )
        == 11.0
    )


@dataclass
class Pair:
    first: float
    second: float


def structured(pair: Pair) -> Pair:
    return Pair(pair.first**2, 3.0 * pair.second)


@ad.rules.jvp_for(structured)
def structured_jvp(tangents: dict[str, object], pair: Pair) -> tuple[Pair, Pair]:
    direction = cast(Pair, tangents["pair"])
    return structured(pair), Pair(
        2.0 * pair.first * direction.first,
        3.0 * direction.second,
    )


def test_jvp_helper_perturbs_dataclasses() -> None:
    assert_jvp_close(
        structured,
        Pair(2.0, 4.0),
        tangents={"pair": Pair(0.5, -1.0)},
    )


def test_real_inner_rejects_incompatible_structures() -> None:
    with pytest.raises(TypeError, match="incompatible"):
        real_inner({"x": 1.0}, 1.0)
    with pytest.raises(TypeError, match="different fields"):
        real_inner({"x": 1.0}, {"y": 1.0})


def wrong_jvp(x: float) -> float:
    return x**2


@ad.rules.jvp_for(wrong_jvp)
def wrong_jvp_rule(tangents: dict[str, object], x: float) -> tuple[float, float]:
    del tangents
    return wrong_jvp(x), 999.0


def test_jvp_helper_reports_all_failed_steps() -> None:
    with pytest.raises(AssertionError, match="JVP finite difference"):
        assert_jvp_close(wrong_jvp, 2.0, tangents={"x": 1.0})


def wrong_vjp(x: float) -> float:
    return x**2


@ad.rules.vjp_for(wrong_vjp)
def wrong_vjp_rule(wrt: tuple[str, ...], x: float) -> tuple[float, object]:
    return wrong_vjp(x), lambda _cotangent: dict.fromkeys(wrt, 999.0)


def test_vjp_helper_reports_all_failed_steps() -> None:
    with pytest.raises(AssertionError, match="VJP finite difference"):
        assert_vjp_close(
            wrong_vjp,
            2.0,
            directions={"x": 1.0},
            cotangent=1.0,
        )


def test_duality_helper_reports_mismatch() -> None:
    @ad.rules.jvp_for(wrong_vjp)
    def wrong_vjp_jvp(tangents: dict[str, object], x: float) -> tuple[float, float]:
        return wrong_vjp(x), 2.0 * x * cast(float, tangents["x"])

    with pytest.raises(AssertionError, match="duality"):
        assert_jvp_vjp_duality(
            wrong_vjp,
            2.0,
            tangents={"x": 1.0},
            cotangent=1.0,
        )
