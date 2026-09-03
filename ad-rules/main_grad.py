import chainrules as ad


def square(x: float) -> float:
    return x * x


@ad.rules.vjp_for(square)
def square_vjp(wrt, x: float):
    value = square(x)

    def pullback(cotangent):
        return {"x": cotangent * 2 * x}

    return value, pullback


gradient = ad.grad(
    square,
    3.0,
    wrt="x",
)

print(gradient)       # {'x': 6.0}
print(gradient["x"])  # 6.0