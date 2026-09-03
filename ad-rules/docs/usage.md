# 使用指南

## 用户：调用已有规则

### JVP

```python
value, tangent_out = ad.jvp(
    solve,
    model,
    params=params,
    tangents={"params": dparams},
)
```

`tangents` 的 key 是原函数参数名。没有出现的参数不活动；`ZERO` 表示零方向。空映射合法，并且不要求已经注册 JVP rule。

### VJP

```python
value, pullback = ad.vjp(
    solve,
    model,
    params=params,
    energy=energy,
    wrt=("params", "energy"),
)

cotangents = pullback(output_cotangent)
grad_params = cotangents["params"]
grad_energy = cotangents["energy"]
```

pullback 总是返回字典，即使 `wrt` 只有一个名字。返回 key 必须与请求完全一致。

### 标量梯度

```python
gradient = ad.grad(loss, params, wrt="params")["params"]
value, gradients = ad.value_and_grad(loss, params, wrt="params")
```

二者只接受单个实标量输出，并使用 cotangent `1.0` 调用 VJP。向量输出需要显式调用 `vjp`，因为“梯度”必须先选择输出方向。

## 规则作者：注册 JVP

```python
@ad.rules.jvp_for(solve)
def solve_jvp(tangents, model, params, controls=None):
    value = solve(model, params, controls)
    dparams = tangents.get("params", ad.ZERO)
    if dparams is ad.ZERO:
        return value, ad.ZERO
    return value, linearized_solve(model, params, dparams, controls)
```

第一个参数固定为 `tangents`，其余调用参数与原 callable 相同。Core 会先验证参数名；rule 负责检查 tangent 的结构、dtype 和支持域。

## 规则作者：注册 VJP

```python
@ad.rules.vjp_for(solve)
def solve_vjp(wrt, model, params, controls=None):
    unsupported = set(wrt) - {"params"}
    if unsupported:
        raise ad.UnsupportedWrt(solve, unsupported, supported={"params"})

    value, state = forward_and_factorize(model, params, controls)

    def pullback(cotangent):
        return {"params": state.parameter_vjp(cotangent)}

    return value, pullback
```

第一个参数固定为规范化后的 `tuple[str, ...]`。只保存 `wrt` 所需状态；不要把未请求梯度全部算完再筛选。

## 嵌套参数

Core 不 flatten 数据，也不解析 `params.gates` 字符串路径。tangent 和 cotangent 应保持 primal 的自然结构：

```python
params = {"gates": gate_values, "zeeman": 0.4}
dparams = {"gates": gate_direction, "zeeman": ad.ZERO}

ad.jvp(f, params, tangents={"params": dparams})
```

叶子级活动裁剪由具体 rule 完成。这样不会迫使所有科学软件接受同一个 pytree 定义。

## 复数

规则应采用实线性约定，把复向量空间视为双倍维度的实向量空间。JVP/VJP 对偶测试使用：

\[
\langle a,b\rangle_{\mathbb R}=\operatorname{Re}(a^*b).
\]

不要在未声明的情况下混用纯复解析导数和实线性梯度。

## 规则包的加载

Core 不自动扫描 entry points。伴随规则包通过显式 import 注册：

```python
import host_package
import host_package_chainrules  # registers rules explicitly
```

显式加载让环境、版本和 import 顺序都可追踪。

## C extension 与实例方法

如果 C extension callable 没有可检查的签名，请提供保留原语义的薄 Python wrapper。实例方法应注册稳定的 unbound callable（如 `Class.method`），避免注册每次属性访问都会生成的新 bound-method 对象。
