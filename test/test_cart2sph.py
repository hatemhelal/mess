import numpy as np
from mess.basis import cart2sph_complex

import pytest
from numpy.testing import assert_allclose

sqrt2 = np.sqrt(2)
sqrt5 = np.sqrt(5)
sqrt3 = np.sqrt(3)

# fmt: off
expect_1 = np.array([
    [1 / sqrt2, 0, 1 / sqrt2],
    [-1j / sqrt2, 0, 1j / sqrt2],
    [0, 1, 0],
])
# fmt: on

# fmt: off
expect_2 = np.array([
    [np.sqrt(3 / 8), 0, -1 / 2, 0, np.sqrt(3 / 8)],
    [-1j / sqrt2, 0, 0, 0, 1j / sqrt2],
    [0, 1 / sqrt2, 0, 1 / sqrt2, 0],
    [-np.sqrt(3 / 8), 0, -1 / 2, 0, -np.sqrt(3 / 8)],
    [0, -1j / sqrt2, 0, 1j / sqrt2, 0],
    [0, 0, 1, 0, 0],
])
# fmt: on


# fmt: off
expect_3 = np.array([
    [sqrt5 / 4, 0, -sqrt3 / 4, 0, -sqrt3 / 4, 0, sqrt5 / 4],
    [-3j / 4, 0, sqrt3 * 1j / (4 * sqrt5), 0, -sqrt3 * 1j / (4 * sqrt5), 0, 3j / 4],
    [0, np.sqrt(3 / 8), 0, -3 / (2 * sqrt5), 0, np.sqrt(3 / 8), 0],
    [-3 / 4, 0, -sqrt3 / (4 * sqrt5), 0, -sqrt3 / (4 * sqrt5), 0, -3 / 4],
    [0, -1j / sqrt2, 0, 0, 0, 1j / sqrt2, 0],
    [0, 0, np.sqrt(3 / 5), 0, np.sqrt(3 / 5), 0, 0],
    [1j * sqrt5 / 4, 0, 1j * sqrt3 / 4, 0, -1j * sqrt3 / 4, 0, -1j * sqrt5 / 4],
    [0, -np.sqrt(3 / 8), 0, -3 / (2 * sqrt5), 0, -np.sqrt(3 / 8), 0],
    [0, 0, -1j * np.sqrt(3 / 5), 0, 1j * np.sqrt(3 / 5), 0, 0],
    [0, 0, 0, 1, 0, 0, 0],
])
# fmt: on


testcases = [
    (0, 1.0),
    (1, expect_1),
    (2, expect_2),
    (3, expect_3),
]


@pytest.mark.parametrize("l,expect", testcases)
def test_cart2sph_complex(l, expect):
    actual = cart2sph_complex(l)
    assert_allclose(actual, expect)
