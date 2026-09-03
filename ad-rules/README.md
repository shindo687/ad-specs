# ChainRules

[![pipeline status](https://git.gewu-lab.ai/kun-agent/chainrules/badges/main/pipeline.svg)](https://git.gewu-lab.ai/kun-agent/chainrules/-/pipelines)
[![coverage report](https://git.gewu-lab.ai/kun-agent/chainrules/badges/main/coverage.svg)](https://git.gewu-lab.ai/kun-agent/chainrules/-/graphs/main/charts)
[![documentation](https://git.gewu-lab.ai/kun-agent/chainrules/badges/main/pipeline.svg?job=documentation)](https://git.gewu-lab.ai/kun-agent/chainrules/-/jobs/artifacts/main/browse/site?job=documentation)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![license: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**A minimal, AD-system-agnostic JVP/VJP rule interface for scientific Python.**

ChainRules 在既有 Python callable 和任意 AD 执行器之间定义一条很窄的协议：规则作者提供高效、可验证的 JVP 或 VJP，上层代码用统一入口调用它们。它不是新的自动微分引擎，不 tracing 任意 Python，也不会在缺少规则时偷偷退回有限差分。

```python
import chainrules as ad

value, dvalue = ad.jvp(
    simulate,
    model,
    params=params,
    tangents={"params": dparams},
)

value, pullback = ad.vjp(
    simulate,
    model,
    params=params,
    wrt="params",
)
grad_params = pullback(output_cotangent)["params"]
```

## 为什么需要这一层

科学软件常已有解析导数、伴随求解器、隐式微分或稀疏线性化，但各自暴露不同的接口。ChainRules 把它们统一为导数线性映射的两种基本作用：

- JVP：给定输入方向，计算 `J @ v`；
- VJP：先做一次 forward，再用可复用 pullback 计算 `J* @ y_bar`。

这条“窄腰”保留原函数、原参数名和原数据结构，不要求把科学模型迁移到另一个对象体系。

## 安装

从工作副本安装：

```bash
python -m pip install .
```

开发环境：

```bash
python -m pip install -e ".[dev,docs]"
```

## 五分钟示例

```python
import chainrules as ad


def energy(x: float, scale: float = 1.0) -> float:
    return scale * x**2


@ad.rules.jvp_for(energy)
def energy_jvp(tangents, x, scale=1.0):
    value = energy(x, scale)
    dx = tangents.get("x", ad.ZERO)
    dscale = tangents.get("scale", ad.ZERO)
    tangent = 0.0
    active = False
    if dx is not ad.ZERO:
        tangent += 2.0 * scale * x * dx
        active = True
    if dscale is not ad.ZERO:
        tangent += x**2 * dscale
        active = True
    return value, tangent if active else ad.ZERO


@ad.rules.vjp_for(energy)
def energy_vjp(wrt, x, scale=1.0):
    unsupported = set(wrt) - {"x", "scale"}
    if unsupported:
        raise ad.UnsupportedWrt(energy, unsupported, supported={"x", "scale"})

    value = energy(x, scale)

    def pullback(cotangent):
        result = {}
        if "x" in wrt:
            result["x"] = cotangent * 2.0 * scale * x
        if "scale" in wrt:
            result["scale"] = cotangent * x**2
        return result

    return value, pullback


assert ad.jvp(energy, 3.0, tangents={"x": 1.0}) == (9.0, 6.0)
assert ad.grad(energy, 3.0, wrt=("x", "scale")) == {
    "x": 6.0,
    "scale": 9.0,
}
```

## 协议保证

- 原 callable 是 primal 语义的唯一来源；注册不会 monkey-patch 原函数。
- `tangents={}` 和全 `ZERO` JVP 不查找规则，也不进入线性化 kernel。
- `pullback(ZERO)` 在 Core 层直接返回所请求输入的 `ZERO`。
- pullback 的 key 必须与 `wrt` 完全相同；缺失和多余都会失败。
- 未注册规则抛出 `RuleNotFound`，不使用生产有限差分 fallback。
- Runtime 没有 NumPy、JAX、PyTorch 或 SciPy 依赖。

## 项目边界

v0.1 有意不提供 tracing、计算图、pytree 系统、完整 Jacobian/Hessian、插件自动发现、后端多重分派、高阶 AD 或公共缓存框架。只有多个独立规则共同证明当前最小协议无法表达必要语义时，才扩展 Core。

完整说明见[文档入口](https://git.gewu-lab.ai/kun-agent/chainrules/-/blob/main/docs/index.md)。CI 会以严格模式自动生成完整 HTML，文档徽章链接到最近一次成功构建的站点 artifact：

- [设计原则](docs/design-principles.md)
- [使用指南](docs/usage.md)
- [最佳实践](docs/best-practices.md)
- [一致性测试](docs/conformance.md)
- [API 参考](docs/reference/api.md)

## 质量门槛

本地与 CI 使用同一组收口命令：

```bash
ruff check .
ruff format --check .
mypy
pytest
mkdocs build --strict
python -m build
```

测试覆盖率门槛为 90%，但覆盖率不是导数正确性的替代品。规则实现还应通过 primal parity、有限差分、多步长扫描、JVP/VJP 对偶、活动输入裁剪、`ZERO` 短路和 pullback 复用测试。

## 状态

`0.1.x` 是最小协议的 alpha 系列。API 变更必须先更新设计不变量和 conformance tests。

## License

[MIT](LICENSE)
