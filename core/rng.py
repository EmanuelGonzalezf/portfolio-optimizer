"""Single randomness API of the project (design.md section 11.1).

Every stochastic component derives its generator from ``child_rng(seed, role)``:
the optimizer multistart ('optimizer'), the weight cloud ('weights'), the
gaussian paths ('paths'), the block bootstrap ('bootstrap') and the sample data
generator ('sample_data'). No function in core/ touches the global ``np.random``
state nor calls ``default_rng()`` without a seed, so two runs with the same seed
are bit-identical.
"""

from __future__ import annotations

import numpy as np

__all__ = ['SEED_ROLES', 'child_rng']

SEED_ROLES = {'weights': 0, 'paths': 1, 'optimizer': 2, 'sample_data': 3, 'bootstrap': 4}


def child_rng(seed: int, role: str) -> np.random.Generator:
    """Return the independent generator assigned to ``role`` under ``seed``.

    The eight-way spawn keeps room for future roles without shifting the streams
    already assigned, so adding a role never changes existing results.
    """
    if role not in SEED_ROLES:
        raise KeyError(role)
    return np.random.default_rng(np.random.SeedSequence(int(seed)).spawn(8)[SEED_ROLES[role]])
