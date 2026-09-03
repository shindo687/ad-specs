# 设计原则

## 1. 原 callable 是唯一的业务接口

规则附着在函数身份上，但不替换函数。注册前后，普通调用始终是：

```python
value = f(*args, **kwargs)
```

因此已有调用者不需要知道 ChainRules，也不会被迫迁移到 `DifferentiableModel`、`ADRequest` 一类平行对象模型。primal 的参数名、默认值和错误行为仍由原函数定义。

## 2. 从导数的线性作用出发

完整 Jacobian 是一种表示，不是求导接口的本质。对于 \(f:\mathbb R^p\to\mathbb R^m\)，Core 只交换：

- JVP：\(Jv\)；
- VJP：\(J^*\bar y\)。

规则可以使用任何内部算法，但不应为了满足协议默认建立 \(m\times p\) 矩阵。这样才能覆盖大参数、稀疏系统和伴随法场景。

## 3. 活动性必须显式

JVP 的 `tangents` 和 VJP 的 `wrt` 是计算预算的一部分，不只是输出筛选器：

```python
ad.jvp(f, x, p, tangents={"p": dp})
ad.vjp(f, x, p, wrt="p")
```

规则只应为被请求的输入建立线性化状态或计算 cotangent。先算全部梯度再丢弃未请求部分，通常违反这条原则。

## 4. 零是控制信号，不是大块存储

`ZERO` 表示 tangent space 中的零元素，但不物化 `zeros_like(primal)`。它允许 Core 在昂贵 kernel 之前短路：

- 空或全 `ZERO` 的 JVP 直接返回 `(f(...), ZERO)`；
- `pullback(ZERO)` 直接返回每个活动输入的 `ZERO`。

Core 不提供 `NONDIFF` 哨兵。没有请求的输入是不活动；请求了不支持的输入抛出 `UnsupportedWrt`；当前点不可微则抛出 `NonDifferentiablePoint`。

## 5. forward 状态属于 pullback

VJP rule 返回 closure：

```python
value, pullback = ad.vjp(f, x, wrt="x")
gx1 = pullback(cotangent_1)
gx2 = pullback(cotangent_2)
```

closure 可以捕获 factorization、checkpoint、残差或其他 forward state。同一个 pullback 多次调用不应重做 primal 或数值分解。Core 不再增加公共 `prepare()` 生命周期，因为 closure 已经表达了必要语义。

## 6. 失败必须比伪成功更明显

Core 从不把“没有解析规则”解释成“使用有限差分”。稳定错误分工如下：

| 情况 | 错误 |
| --- | --- |
| 没注册所需模式 | `RuleNotFound` |
| 规则不支持请求的输入 | `UnsupportedWrt` |
| 支持域内的一般规则在当前点失效 | `NonDifferentiablePoint` |

阈值、秩变化、拓扑变化、未收敛和奇异线性化都应显式报告，而不是返回零或未经声明的近似。

## 7. Pythonic 意味着尊重 Python

- 使用 `inspect.signature` 和参数名，而不是脆弱的整数位置索引；
- tangent 保持原生 tuple、list、dict、dataclass 或数组结构；
- 显式 import 规则包完成注册，不扫描全环境；
- decorator 返回原 rule，便于测试和调试；
- 重复注册立即失败，避免 import 顺序静默改变行为。

## 8. 有意留下的空白

v0.1 不包含 tracing、pytree、插件发现、多后端分派、高阶 AD、完整 Jacobian/Hessian、公共缓存协议或版本约束语言。这不是遗漏，而是扩展纪律：只有多个独立适配共同遇到同一个不可表达问题，才把解决方案提升到 Core。
