# Contributing

Public API changes start with the protocol invariants, not with implementation convenience. A proposal should identify at least two independent rule providers that cannot express a required behavior with the current callable/JVP/VJP/closure model.

Before opening a merge request, run:

```bash
ruff check .
ruff format --check .
mypy
pytest
mkdocs build --strict
python -m build
```

New rules should include primal parity, multi-step finite-difference, JVP/VJP duality, activity pruning, zero short-circuit, pullback reuse, and boundary-failure tests as applicable.
