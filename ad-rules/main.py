import chainrules as ad


# 原来的科学函数，不需要修改
def square(x: float) -> float:
    return x * x


# 给 square 注册一个 JVP 规则
@ad.rules.jvp_for(square)
def square_jvp(tangents, x: float):
    value = square(x)

    dx = tangents.get("x", ad.ZERO)
    if dx is ad.ZERO:
        return value, ad.ZERO

    # d(x²) = 2x·dx
    return value, 2 * x * dx


value, tangent = ad.jvp(
    square,
    4.0,
    tangents={"x": 1.0},
)

print(value)
print(tangent)