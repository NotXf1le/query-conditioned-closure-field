from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from grounding_permutation_diagnostics import (
    canonical_accuracy,
    mutual_information_from_confusion,
    permutation_aligned_accuracy,
)


def test_permutation_alignment_recovers_permuted_relation_code() -> None:
    conf = torch.zeros(4, 4, dtype=torch.long)
    perm = [2, 0, 3, 1]
    for true, pred in enumerate(perm):
        conf[true, pred] = 5

    canonical = canonical_accuracy(conf)
    aligned, best_perm = permutation_aligned_accuracy(conf)

    assert canonical == 0.0
    assert aligned == 1.0
    assert best_perm == perm
    assert mutual_information_from_confusion(conf) > 0.99


def test_permutation_alignment_handles_empty_confusion() -> None:
    conf = torch.zeros(3, 3, dtype=torch.long)
    aligned, best_perm = permutation_aligned_accuracy(conf)

    assert math.isnan(aligned)
    assert best_perm == []
