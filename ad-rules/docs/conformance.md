# 一致性测试

`chainrules.testing` 提供依赖无关的测试 helper。有限差分只存在于这个测试模块，不会参与生产调度。

```python
from chainrules.testing import (
    assert_jvp_close,
    assert_jvp_vjp_duality,
    assert_vjp_close,
)

assert_jvp_close(
    energy,
    3.0,
    tangents={"x": 0.25},
)

assert_vjp_close(
    energy,
    3.0,
    directions={"x": 0.25},
    cotangent=2.0,
)

assert_jvp_vjp_duality(
    energy,
    3.0,
    tangents={"x": 0.25},
    cotangent=2.0,
)
```

## 为什么扫描多个步长

单个有限差分步长可能偶然通过：步长过大时截断误差主导，过小时舍入误差或 solver 噪声主导。helper 默认检查 `1e-4`、`1e-5`、`1e-6`，任一合理步长满足容差即可。规则包应按实际数值尺度调整扫描范围，而不是放宽到失去判别力。

## 为什么还要测对偶

JVP 和 VJP 可能分别对某个有限差分 fixture 通过，却采用了不同的复数约定、归一化或参数排序。对偶关系

\[
\langle Jv,\bar y\rangle_{\mathbb R}
=
\langle v,J^*\bar y\rangle_{\mathbb R}
\]

直接约束两者必须描述同一个线性映射，而且不需要构造完整 Jacobian。

## 性能性质不能用有限差分证明

下面的性质应通过计数器断言：

- 全零 JVP 不进入 JVP rule；
- `pullback(ZERO)` 不进入 raw pullback；
- 只请求 `params` 时不执行其他输入的 derivative kernel；
- 重复调用 pullback 不重复 primal 和 factorization。

墙钟 benchmark 可以补充观察，但容易受机器噪声影响，不能替代结构化调用计数。
