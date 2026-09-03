# 与 Julia ChainRules 生态的关系

本项目采用简洁的 Python 包名 `chainrules`。它受 Julia
ChainRulesCore 的分层思想启发，但不是 Julia API 的逐项移植，也不声称与
Julia ChainRules 规则集合兼容。

Julia 生态把规则协议放在 ChainRulesCore，把 Julia Base 和标准库的具体规则
放在 ChainRules。本项目 v0.1 只负责前一层：统一规则接口、注册与调度；具体
科学软件的规则仍由独立适配包提供。只有出现足够多、真正跨软件复用的规则时，
才考虑在本项目中维护公共规则集合。

| 能力 | ChainRules v0.1 | ChainRulesCore |
| --- | --- | --- |
| 原函数之外定义规则 | 是 | 是 |
| Forward rule | JVP rule | `frule` |
| Reverse rule | VJP + pullback | `rrule` + pullback |
| 零 tangent | `ZERO` | `ZeroTangent` |
| 活动输入选择 | 参数名 `wrt` / `tangents` | 调用约定与 tangent 元组 |
| 规则分派 | callable identity registry | Julia 多重分派 |
| 结构化 tangent 类型 | 使用原生 Python 结构 | `Tangent` 等完整体系 |
| 不可微/无 tangent 区分 | 统一通过活动性和错误表达 | `NoTangent` / `ZeroTangent` |
| 延迟 cotangent | pullback closure 内部实现 | `Thunk` / `InplaceableThunk` |
| 配置化规则 | 暂无 | `RuleConfig` |
| 高阶与复杂投影支持 | 暂无 | 更完整 |

因此，v0.1 可以称为 **ChainRulesCore-lite 的 Python 窄腰**：数学核心相同，工程表面更小。包名统一为 `chainrules`，不改变这条职责边界。

缺少的能力不是被否认，而是尚未获得进入公共接口的证据。Python 缺少 Julia 的多重分派；直接模仿所有 tangent 类型、thunk 和配置机制，很容易先建立框架、后寻找需求。ChainRules 采用相反次序：先让多个真实科学规则共享最小 JVP/VJP 协议，只有重复出现的不可表达需求才触发扩展。
