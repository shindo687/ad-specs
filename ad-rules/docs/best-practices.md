# 最佳实践

## 先定义支持域，再写公式

规则文档应明确：支持哪些输入、dtype、输出结构和物理域；哪些阈值或离散变化不可微。支持域不是事后注释，而是规则正确性的一部分。

## 复用 forward，而不是验证时才想起 parity

理想实现是在原 host kernel 上增加 derivative hook，共用一次 forward。若必须重建数值路径，应建立独立 primal-parity gate，覆盖多个代表性 fixture。生产环境不必每次运行两套 solver，但验收阶段必须证明语义一致。

## 让 `wrt` 真正降低成本

对于标量目标和大量参数，VJP 的昂贵求解次数不应随参数数目线性增长。用 kernel 计数器测试这个性质，比只比较墙钟时间更可靠。

推荐记录：

- primal 调用次数；
- factorization 次数；
- forward/adjoint solve 次数；
- 每个活动输入专属 kernel 的调用次数。

## pullback 必须可复用

```python
value, pullback = ad.vjp(f, x, wrt="x")
g1 = pullback(c1)
g2 = pullback(c2)
```

第二次调用不应增加 primal 或 factorization 计数。若底层算法只能一次性消费状态，应在规则层复制轻量状态或明确失败；不要静默返回错误结果。

## `ZERO` 短路要发生在昂贵 kernel 之前

Core 已经短路全零 JVP 和零 cotangent pullback。规则仍需处理“多个 tangent 中只有一部分为零”的情况，避免为零方向启动线性求解。

## 不要隐藏有限差分

有限差分适合测试 oracle 和 benchmark comparator，不适合生产 fallback。隐藏 fallback 会把“不支持”伪装成“支持”，同时引入步长、噪声和复杂度风险。

## 不要注册临时 callable

Registry 以 callable identity 为 key。注册模块级函数、稳定 wrapper 或 unbound method。不要注册 lambda 后丢失引用，也不要注册一次属性访问产生的 bound method，再用另一次属性访问调用。

## 错误要包含物理原因

在 `NonDifferentiablePoint` 的消息中写明当前点失败的原因，例如“本征值在容差内简并”“Newton solver 未收敛”或“active set changed”。底层数值异常可以通过 `raise ... from error` 保留为 cause。

## 每个规则的最低测试矩阵

1. 原函数与 rule primal 一致；
2. JVP 与多步长中心差分一致；
3. VJP 与标量化方向差分一致；
4. JVP/VJP 满足实内积对偶；
5. 全零和部分零方向短路；
6. 只计算请求的活动输入；
7. pullback 多次调用不重复 forward；
8. 已知非光滑边界显式失败；
9. 支持的 dtype 和嵌套结构有固定 fixture；
10. 源码与运行路径不存在通用有限差分 fallback。

## 何时应该扩展 Core

一个适配包需要额外 helper，不足以扩展公共协议。至少等待两个相互独立的规则实现出现同一种无法在现有 JVP/VJP、closure、原生 Python 结构和显式错误中表达的需求，再讨论新概念。
