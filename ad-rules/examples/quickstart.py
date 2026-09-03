"""Executable version of the README quickstart."""

import chainrules as ad


def energy(x: float, scale: float = 1.0) -> float:
    """Return a scalar quadratic energy."""
    return scale * x**2


@ad.rules.jvp_for(energy)
def energy_jvp(
    tangents: dict[str, object], x: float, scale: float = 1.0
) -> tuple[float, object]:
    """Compute the directional derivative of ``energy``."""
    value = energy(x, scale)
    dx = tangents.get("x", ad.ZERO)
    dscale = tangents.get("scale", ad.ZERO)
    tangent = 0.0
    active = False
    if dx is not ad.ZERO:
        tangent += 2.0 * scale * x * float(dx)  # type: ignore[arg-type]
        active = True
    if dscale is not ad.ZERO:
        tangent += x**2 * float(dscale)  # type: ignore[arg-type]
        active = True
    return value, tangent if active else ad.ZERO


@ad.rules.vjp_for(energy)
def energy_vjp(
    wrt: tuple[str, ...], x: float, scale: float = 1.0
) -> tuple[float, object]:
    """Create a reusable pullback for ``energy``."""
    unsupported = set(wrt) - {"x", "scale"}
    if unsupported:
        raise ad.UnsupportedWrt(energy, unsupported, supported={"x", "scale"})
    value = energy(x, scale)

    def pullback(cotangent: object) -> dict[str, object]:
        seed = float(cotangent)  # type: ignore[arg-type]
        result: dict[str, object] = {}
        if "x" in wrt:
            result["x"] = seed * 2.0 * scale * x
        if "scale" in wrt:
            result["scale"] = seed * x**2
        return result

    return value, pullback


value, tangent = ad.jvp(energy, 3.0, tangents={"x": 1.0})
assert (value, tangent) == (9.0, 6.0)

gradient = ad.grad(energy, 3.0, wrt=("x", "scale"))
assert gradient == {"x": 6.0, "scale": 9.0}
