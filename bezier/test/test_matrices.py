import pytest
from bezier.matrices import binom, get_pascals, get_mix_matrix
import numpy as np


class TestGetPascals:
    @pytest.mark.parametrize(
        "in_out", [(1, 1), (2, 1, 1), (3, 1, 2, 1), (4, 1, 3, 3, 1), (5, 1, 4, 6, 4, 1)]
    )
    def test_binom(self, in_out) -> None:
        """ Test against known values """
        assert tuple(get_pascals(in_out[0])) == in_out[1:]

class TestGetMixmat:
    def test_get_mixmat(self):
        get_mix_matrix(4)
