# ChainRules：让科学软件共享同一种求导语言

科学软件的导数通常不是“没有”，而是散落在解析公式、伴随程序、隐式求解器和稀疏线性化代码中。真正缺少的是一条共同协议：上层系统如何在不改写原 API 的前提下调用这些导数？

ChainRules 的回答刻意很小：

> 保留原 Python callable，只为它注册 JVP 和 VJP；其他一阶便利接口都从这两者派生。

```python
import chainrules as ad

value, tangent = ad.jvp(f, x, tangents={"x": dx})
value, pullback = ad.vjp(f, x, wrt="x")
gradient = pullback(cotangent)["x"]
```

## 一条窄腰，而不是另一个框架

接口的上方可以是优化器、反问题、参数估计或某个 AD engine；接口的下方可以是解析导数、PDE 伴随、隐式微分、外部 Fortran/C++ kernel。Core 不知道这些算法，只固定交换语义：

| 层次 | 责任 |
| --- | --- |
| 上层执行器 | 组织计算、选择 forward/reverse mode、消费规则 |
| **ChainRules** | 注册、参数活动性、JVP/VJP 调度、零短路、错误语义 |
| 科学软件/规则包 | primal、线性化算法、状态复用、支持域与数值有效性 |

这种分层的价值是稳定性。领域代码继续使用熟悉的 `f(*args, **kwargs)`；规则是附加能力，不是第二套业务 API。

## 第一性原理

在点 \(x\)，导数是线性映射 \(Df(x)\)。绝大多数一阶任务只需要它的两种作用，而不需要完整 Jacobian：

\[
\dot y = Df(x)\dot x,
\qquad
\bar x = Df(x)^*\bar y.
\]

前者是 JVP，适合少量输入方向；后者是 VJP，适合标量目标对大量参数的梯度。`grad` 只是给 VJP 的 pullback 输入 `1.0`，并不是第三种规则。

## 最小公共面

用户入口只有：

- `jvp`、`vjp`；
- 从 VJP 派生的 `grad`、`value_and_grad`；
- `ZERO`；
- `rules.jvp_for`、`rules.vjp_for`；
- 三种稳定错误：`RuleNotFound`、`UnsupportedWrt`、`NonDifferentiablePoint`。

下一步：阅读[设计原则](design-principles.md)，或直接进入[使用指南](usage.md)。
