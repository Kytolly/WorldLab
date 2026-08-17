from __future__ import annotations

import numpy as np
import pytest

from worldlab.core import ArraySpace, DictSpace, DiscreteSpace, TupleSpace


def test_array_space_supports_optional_bounds() -> None:
    space = ArraySpace((2,), dtype=np.float32, low=-1.0, high=1.0)
    sample = space.sample()

    assert sample.shape == (2,)
    assert space.contains(sample)
    assert space.contains(np.array([0.0, 1.0], dtype=np.float64))
    assert not space.contains(np.array([0.0, 1.1], dtype=np.float32))


def test_array_space_integer_dtype_keeps_unbounded_default_compatible() -> None:
    space = ArraySpace((2,), dtype=np.int32)
    assert space.contains(space.sample())


def test_nested_spaces_sample_contains_and_seed_recursively() -> None:
    def make_space() -> DictSpace:
        return DictSpace(
            {
                "policy": DictSpace(
                    {
                        "state": ArraySpace((2,), dtype=np.float32, low=-1.0, high=1.0),
                        "mode": DiscreteSpace(3),
                    }
                ),
                "critic": TupleSpace(
                    (ArraySpace((1,), dtype=np.float32), DiscreteSpace(2))
                ),
            }
        )

    first = make_space()
    second = make_space()
    first.seed(7)
    second.seed(7)
    first_sample = first.sample()
    second_sample = second.sample()

    assert np.array_equal(first_sample["policy"]["state"], second_sample["policy"]["state"])
    assert first_sample["policy"]["mode"] == second_sample["policy"]["mode"]
    assert np.array_equal(first_sample["critic"][0], second_sample["critic"][0])
    assert first_sample["critic"][1] == second_sample["critic"][1]
    assert first.contains(first_sample)

    missing = dict(first_sample)
    del missing["critic"]
    assert not first.contains(missing)
    assert not first.contains({**first_sample, "extra": 0})
    assert not first.contains({"policy": first_sample["policy"], "critic": list(first_sample["critic"])})


def test_spaces_reject_invalid_children_and_keys() -> None:
    with pytest.raises(TypeError, match="implement"):
        DictSpace({"bad": object()})
    with pytest.raises(ValueError, match="non-empty strings"):
        DictSpace({"": DiscreteSpace(2)})
