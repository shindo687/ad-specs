import chainrules as ad


def square(x: float) -> float:
    return x * x


@ad.rules.vjp_for(square)
def square_vjp(wrt, x: float):
    value = square(x)

    def pullback(cotangent):
        # square'(x) = 2x
        return {"x": cotangent * 2 * x}

    return value, pullback


value, pullback = ad.vjp(
    square,
    3.0,
    wrt="x",
)

print(value)             # 9.0
print(pullback(1.0))     # {'x': 6.0}
print(pullback(2.0))     # {'x': 12.0}