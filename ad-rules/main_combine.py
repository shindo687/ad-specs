import chainrules as ad


# ---------- 第一层：f(x) = x² ----------

def f(x: float) -> float:
    return x * x


@ad.rules.jvp_for(f)
def f_jvp(tangents, x: float):
    dx = tangents.get("x", ad.ZERO)

    if dx is ad.ZERO:
        return f(x), ad.ZERO

    return f(x), 2 * x * dx


@ad.rules.vjp_for(f)
def f_vjp(wrt, x: float):
    def pullback(cotangent):
        return {
            "x": cotangent * 2 * x,
        }

    return f(x), pullback


# ---------- 第二层：g(y) = 3y ----------

def g(y: float) -> float:
    return 3 * y


@ad.rules.jvp_for(g)
def g_jvp(tangents, y: float):
    dy = tangents.get("y", ad.ZERO)

    if dy is ad.ZERO:
        return g(y), ad.ZERO

    return g(y), 3 * dy


@ad.rules.vjp_for(g)
def g_vjp(wrt, y: float):
    def pullback(cotangent):
        return {
            "y": cotangent * 3,
        }

    return g(y), pullback


# ---------- 组合：h(x) = g(f(x)) ----------

def h(x: float) -> float:
    return g(f(x))


@ad.rules.jvp_for(h)
def h_jvp(tangents, x: float):
    # 先经过 f：x -> y
    y, dy = ad.jvp(
        f,
        x,
        tangents=tangents,
    )

    # 再经过 g：y -> z
    return ad.jvp(
        g,
        y,
        tangents={"y": dy},
    )


@ad.rules.vjp_for(h)
def h_vjp(wrt, x: float):
    # 正向得到 f 的 pullback
    y, pullback_f = ad.vjp(
        f,
        x,
        wrt=wrt,
    )

    # 正向得到 g 的 pullback
    value, pullback_g = ad.vjp(
        g,
        y,
        wrt="y",
    )

    def pullback(cotangent):
        # 先从 g 的输出 z 反传到 y
        dy = pullback_g(cotangent)["y"]

        # 再从 y 反传到 x
        return pullback_f(dy)

    return value, pullback


# ---------- 使用 ----------

print("普通调用：", h(2.0))

print(
    "JVP：",
    ad.jvp(
        h,
        2.0,
        tangents={"x": 1.0},
    ),
)

value, pullback = ad.vjp(
    h,
    2.0,
    wrt="x",
)

print("VJP 的函数值：", value)
print("VJP 的梯度：", pullback(1.0))

print(
    "grad：",
    ad.grad(
        h,
        2.0,
        wrt="x",
    ),
)