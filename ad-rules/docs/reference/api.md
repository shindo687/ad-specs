# API 参考

## 顶层接口

::: chainrules
    options:
      members:
        - ZERO
        - jvp
        - vjp
        - grad
        - value_and_grad
        - rules
        - RuleRegistry
        - RuleNotFound
        - UnsupportedWrt
        - NonDifferentiablePoint

## Registry

::: chainrules.registry.RuleRegistry

## 一致性测试工具

::: chainrules.testing
    options:
      members:
        - real_inner
        - assert_jvp_close
        - assert_vjp_close
        - assert_jvp_vjp_duality
