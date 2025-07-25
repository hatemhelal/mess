"""Functions for constructing Gaussian orbital shells."""

from collections import deque
from functools import cache
from typing import List

import numpy as np

from mess.orbital import Orbital, make_contraction
from mess.types import Float3, FloatN


@cache
def cart_lmn(L: int) -> List[tuple[int, int, int]]:
    """Generates Cartesian angular momentum quantum numbers (lx, ly, lz) for a given L.

    The function returns a list of all non-negative integer triplets (lx, ly, lz)
    such that their sum equals the total angular momentum L. The list is sorted
    in descending order of lx, then ly, then lz. This ordering is consistent
    with common quantum chemistry packages.

    Args:
        L (int): The total angular momentum quantum number.

    Returns:
        List[tuple[int, int, int]]: A list of (lx, ly, lz) tuples.
    """
    if L < 0:
        return []
    lmns = []
    for lx in range(L, -1, -1):
        for ly in range(L - lx, -1, -1):
            lz = L - lx - ly
            lmns.append((lx, ly, lz))
    return lmns


def make_shell(
    spherical: bool,
    L: int,
    center: Float3,
    alphas: FloatN,
    coefficients: FloatN,
) -> List[Orbital]:
    """Constructs a list of `Orbital` objects representing a Gaussian shell.

    A Gaussian shell is a set of orbitals sharing the same total angular momentum `L`.
    This function creates either Cartesian or spherical Gaussian shells based on the
    `spherical` flag. For Cartesian shells, it generates an `Orbital` for each
    (lx, ly, lz) combination corresponding to the total angular momentum L.
    For spherical shells, it uses transformation matrices to convert Cartesian
    primitives into real spherical harmonics.

    Args:
        spherical (bool): If True, constructs spherical Gaussian shells.
                          Otherwise, constructs Cartesian Gaussian shells.
        L (int): The total angular momentum quantum number of the shell.
        center (Float3): The common center of all primitives in the shell.
        alphas (FloatN): Array of exponent values for each primitive.
        coefficients (FloatN): Array of contraction coefficients for each primitive.

    Returns:
        List[Orbital]: A list of Orbital objects forming the specified Gaussian shell.
    """

    if spherical and L > 1:
        return make_spherical_shell(L, center, alphas, coefficients)

    out = []

    for lmn in cart_lmn(L):
        lmn = np.tile(np.array(lmn, dtype=np.int32), (len(alphas), 1))
        ao = make_contraction(center, alphas, lmn, coefficients)
        out.append(ao)

    return out


def make_spherical_shell(L, center, alphas, coefficients):
    """Constructs a list of Orbital objects representing a spherical Gaussian shell.

    This function takes Cartesian primitives and transforms them into real spherical
    harmonics using pre-calculated transformation matrices. It then creates `Orbital`
    objects for each resulting spherical component.

    Note that broadcasting is applied to ensure that `alphas` and `coefficients` are
    correctly expanded to compose the linear combination necessary for the spherical
    Gaussian format.

    Args:
        L (int): The total angular momentum quantum number of the shell.
        center (Float3): The common center of all primitives in the shell.
        alphas (FloatN): Array of exponent values for each primitive.
        coefficients (FloatN): Array of contraction coefficients for each primitive.

    Returns:
        List[Orbital]: A list of Orbital objects forming the spherical Gaussian shell.
    """

    out = []
    C = cart2sph_real(L)
    cart_idx, sph_idx = np.nonzero(C)
    lmn_map = cart_lmn(L)

    for idx in range(C.shape[1]):
        lmns = [lmn_map[lmn_idx] for lmn_idx in cart_idx[sph_idx == idx]]
        sph_c = C[cart_idx[sph_idx == idx], idx]
        num_cart_components = len(lmns)
        num_primitives = len(alphas)

        lmns = np.tile(np.array(lmns, dtype=np.int32), (num_primitives, 1))
        sph_c = coefficients[:, None] * sph_c[None, :]
        sph_alphas = np.repeat(alphas, num_cart_components)

        ao = make_contraction(center, sph_alphas, lmns, sph_c.reshape(-1))
        out.append(ao)

    return out


def cart2sph_coef(lmn: tuple[int, int, int], l: int, m: int) -> complex:
    """Transformation coefficients for Cartesian to spherical Gaussian basis functions.

    This function calculates the transformation coefficient from a Cartesian Gaussian
    function (defined by `lmn`) to a spherical Gaussian function (defined by `l` and
    `m`). The formula used is based on the relationship between Cartesian and spherical
    harmonics. See equation 15 of <https://doi.org/10.1002/qua.560540202>.


    Args:
        lmn (tuple): A tuple (lx, ly, lz) representing the angular momentum
                     components of the Cartesian Gaussian function.
        l (int): The total angular momentum quantum number of the spherical harmonic.
        m (int): The magnetic quantum number of the spherical harmonic.

    Returns:
        complex: The transformation coefficient as a complex scalar
    """
    from scipy.special import binom, factorial

    lmn, l, m = np.array(lmn), np.array(l), np.array(m)

    # Check if j = (lx + ly - |m|) / 2 is an integer
    abs_m = np.abs(m)
    j = lmn[0] + lmn[1] - abs_m

    if j % 2 != 0:
        # j must be half-integral so coefficient must be zero
        return np.array(0.0, dtype=complex)
    else:
        j = j // 2

    # constant pre-factor
    num = np.prod(factorial(2 * lmn)) * factorial(l) * factorial(l - abs_m)
    dem = factorial(2 * l) * np.prod(factorial(lmn)) * factorial(l + abs_m)
    out = np.sqrt(num / dem) / (2**l * factorial(l))

    # sum_i taking into account that binom(p, q) is zero for q < 0 and q > p
    # as well as avoiding negative factorial in the denominator
    i = np.arange(np.maximum(j, 0), (l - abs_m) // 2 + 1)
    iterm_num = binom(l, i) * binom(i, j) * (-1) ** i * factorial(2 * l - 2 * i)
    out *= np.sum(iterm_num / factorial(l - abs_m - 2 * i))

    k = np.arange(0, j + 1)
    if len(k) > 0:
        kterm = binom(j, k) * binom(abs_m, lmn[0] - 2 * k)
        power = np.sign(m) * 0.5 * (abs_m - lmn[0] + 2 * k)
        out *= np.sum(kterm * np.power(-1.0, power, dtype=complex))

    return out.astype(complex)


@cache
def cart2sph_complex(l: int) -> np.ndarray:
    """Transformation matrix for Cartesian to spherical Gaussian coefficients.

    This function generates a transformation matrix that converts Cartesian Gaussian
    coefficients to spherical Gaussian coefficients for a given total angular momentum
    `l`. Each row of the matrix corresponds to a Cartesian basis function (defined by
    `lmn` from `LMN_MAP[l]`), and each column corresponds to a spherical basis function
    (defined by `m` from `-l` to `l`).

    Args:
        l (int): The total angular momentum quantum number.

    Returns:
        np.ndarray: A 2D NumPy array representing the transformation matrix.
                    The shape of the matrix is `(num_cartesian_functions, 2*l + 1)`.
    """

    return np.asarray([
        [cart2sph_coef(lmn, l, m) for m in range(-l, l + 1)] for lmn in cart_lmn(l)
    ])


@cache
def cart2sph_real(l: int) -> np.ndarray:
    """Transformation matrix for Cartesian to real spherical Gaussian coefficients.

    This function generates a transformation matrix that converts Cartesian Gaussian
    coefficients to real spherical Gaussian coefficients for a given total angular
    momentum `l`. Each row of the matrix corresponds to a Cartesian basis function
    (defined by `lmn` from `LMN_MAP[l]`), and each column corresponds to a real
    spherical basis function (defined by `m` from `-l` to `l`, ordered as
    -l, ..., -1, 0, 1, ..., l).

    Args:
        l (int): The total angular momentum quantum number.

    Returns:
        np.ndarray: A 2D NumPy array representing the transformation matrix.
                    The shape of the matrix is `(num_cartesian_functions, 2*l + 1)`.
    """

    out = deque()
    lmn_map = cart_lmn(l)
    t0 = np.array([cart2sph_coef(lmn, l, 0) for lmn in lmn_map]).real
    out.append(t0)

    for m in range(1, l + 1):
        m_pos = np.array([cart2sph_coef(lmn, l, m) for lmn in lmn_map])
        m_neg = np.array([cart2sph_coef(lmn, l, -m) for lmn in lmn_map])
        plus = (m_neg + m_pos) / np.sqrt(2)
        minus = (m_neg - m_pos) * 1j / np.sqrt(2)
        out.appendleft(np.real_if_close(minus))
        out.append(np.real_if_close(plus))

    # transpose so leading dim is in cartesian basis and clip small elements to zero
    out = np.array(out).T
    eps = np.finfo(out.dtype).eps
    out[np.abs(out) < eps] = 0.0
    return out
