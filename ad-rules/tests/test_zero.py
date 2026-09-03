from __future__ import annotations

import copy

from chainrules import ZERO


def test_zero_is_readable_and_copy_stable() -> None:
    assert repr(ZERO) == "ZERO"
    assert copy.copy(ZERO) is ZERO
    assert copy.deepcopy(ZERO) is ZERO
