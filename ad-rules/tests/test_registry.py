from __future__ import annotations

from collections.abc import Callable

import pytest

from chainrules import RuleNotFound, RuleRegistry


def original(x: float) -> float:
    return x


def jvp_rule(tangents: object, x: float) -> tuple[float, float]:
    del tangents
    return x, 1.0


def vjp_rule(wrt: object, x: float) -> tuple[float, Callable[[float], object]]:
    del wrt
    return x, lambda cotangent: {"x": cotangent}


def test_separate_registry_registers_and_retrieves_both_modes() -> None:
    registry = RuleRegistry()
    assert registry.jvp_for(original)(jvp_rule) is jvp_rule
    assert registry.vjp_for(original)(vjp_rule) is vjp_rule
    assert registry.get_jvp(original) is jvp_rule
    assert registry.get_vjp(original) is vjp_rule


@pytest.mark.parametrize("mode", ["jvp", "vjp"])
def test_missing_rule_has_stable_error(mode: str) -> None:
    registry = RuleRegistry()
    getter = registry.get_jvp if mode == "jvp" else registry.get_vjp
    with pytest.raises(RuleNotFound, match=f"No {mode.upper()} rule") as captured:
        getter(original)
    assert captured.value.function is original
    assert captured.value.mode == mode.upper()


@pytest.mark.parametrize("mode", ["jvp", "vjp"])
def test_duplicate_registration_is_rejected(mode: str) -> None:
    registry = RuleRegistry()
    decorator = (
        registry.jvp_for(original) if mode == "jvp" else registry.vjp_for(original)
    )
    decorator(jvp_rule)
    with pytest.raises(RuntimeError, match="already registered"):
        decorator(vjp_rule)


class UnhashableCallable:
    __hash__ = None  # type: ignore[assignment]

    def __call__(self, x: float) -> float:
        return x


def test_registry_uses_identity_even_for_unhashable_callables() -> None:
    first = UnhashableCallable()
    second = UnhashableCallable()
    registry = RuleRegistry()
    registry.jvp_for(first)(jvp_rule)
    assert registry.get_jvp(first) is jvp_rule
    with pytest.raises(RuleNotFound):
        registry.get_jvp(second)
