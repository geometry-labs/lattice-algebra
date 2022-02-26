"""
Tests the lattices package.

Todo list:
 1. Test to_bits and to_bytes for both Polynomial and PolynomialVector
 2. Test hash functions
"""
import pytest
from lattice_algebra.main import *
from copy import deepcopy
from secrets import randbelow, randbits
from typing import List, Tuple, Dict

# Test runtime is O(3 minutes + sample_size * 20 seconds) on my desktop
sample_size_for_random_tests: int = 2 ** 3

small_q_for_testing: int = 17
small_non_prime_for_testing: int = 16
small_d_for_testing: int = 8
small_non_ntt_prime_for_testing: int = 11
allowed_primitive_roots_of_unity_for_small_q_and_small_d: List[int] = [3, 5, 6, 7, 10, 11, 12, 14]
small_halfmod_for_testing: int = small_q_for_testing // 2
small_logmod_for_testing: int = ceil(log2(small_q_for_testing))
small_n_for_testing: int = 2 * small_d_for_testing
small_lgn_for_testing: int = ceil(log2(small_n_for_testing))

modulus_for_testing: int = 40961
degree_for_testing: int = 4096
length_for_testing: int = 3
norm_for_testing: int = 1
weight_for_testing: int = 2
halfmod_for_testing: int = modulus_for_testing // 2
logmod_for_testing: int = ceil(log2(modulus_for_testing))
n_for_testing: int = 2 * degree_for_testing
lgn_for_testing: int = ceil(log2(n_for_testing))

pars_for_testing: dict = {
    'modulus': modulus_for_testing, 'degree': degree_for_testing, 'length': length_for_testing,
    'halfmod': halfmod_for_testing, 'logmod': logmod_for_testing, 'n': n_for_testing,
    'lgn': lgn_for_testing
}
lp_for_testing = LatticeParameters(
    degree=pars_for_testing['degree'], length=pars_for_testing['length'], modulus=pars_for_testing['modulus']
)
secpar4testing = 8

IS_PRIME_CASES = [
    (17, True),
    (8675309, True),
    (16, False),
    (small_q_for_testing, True),
    (small_q_for_testing + 1, False),
    (modulus_for_testing, True),
    (modulus_for_testing - 1, False)
]


# @pytest.mark.skip
@pytest.mark.parametrize("q,expected_output", IS_PRIME_CASES)
def test_is_prime(q, expected_output):
    assert is_prime(q=q) == expected_output


IS_POW_TWO_CASES = [
    (2, True),
    (4, True),
    (8, True),
    (16, True),
    (3, False),
    (5, False),
    (9, False),
    (17, False)
]


@pytest.mark.parametrize("d,expected_output", IS_POW_TWO_CASES)
def test_is_pow_two(d, expected_output):
    assert is_pow_two(val=d) == expected_output


HAS_PRIM_ROU_CASES = [
    (17, 8, True),
    (18, 8, False),
    (33, 8, True),
    (34, 8, False),
    (257, 64, True),  # SWIFFT parameters
    (258, 64, False),
    (8380417, 256, True),  # CRYSTALS-Dilithium
    (8380418, 256, False),
    (201, 25, True),  # We don't need the primality of q or power-of-two d
]


@pytest.mark.parametrize("q,d,expected_output", HAS_PRIM_ROU_CASES)
def test_has_prim_rou(q, d, expected_output):
    assert has_prim_rou(q=q, d=d) == expected_output


NTT_FRIENDLY_CASES = [
    (True, True, True, True),
    (True, True, False, False),
    (True, False, True, False),
    (False, True, True, False),
    (True, False, False, False),
    (False, True, False, False),
    (False, False, True, False),
    (False, False, False, False),
]


@pytest.mark.parametrize("foo,bar,baz,expected_output", NTT_FRIENDLY_CASES)
def test_is_ntt_friendly_prime_with_mock(mocker, foo, bar, baz, expected_output):
    mocker.patch('lattice_algebra.main.is_prime', return_value=foo)
    mocker.patch('lattice_algebra.main.is_pow_two', return_value=bar)
    mocker.patch('lattice_algebra.main.has_prim_rou', return_value=baz)
    assert is_ntt_friendly_prime(q=1, d=1) == expected_output  # the actual input doesn't matter


IS_PRIM_ROU_CASES = [
    (17, 8, 3, True),
    (17, 8, 4, False),
    (17, 8, 5, True),
    (17, 8, 6, True),
    (17, 8, 7, True),
    (17, 8, 8, False),
    (17, 8, 9, False),
    (17, 8, 10, True),
    (17, 8, 11, True),
    (17, 8, 12, True),
    (17, 8, 13, False),
    (17, 8, 14, True),
    (17, 8, 15, False),
    (17, 8, 16, False),
    (8380417, 256, 1753, True),  # CRYSTALS-Dilithium
    (8380417, 257, 1753, False),
    (8380418, 256, 1753, False),
    (257, 64, 42, True),  # SWIFFT
    (257, 65, 42, False),
    (258, 64, 42, False),
]


@pytest.mark.parametrize("q,d,i,expected_output", IS_PRIM_ROU_CASES)
def test_is_prim_rou(q, d, i, expected_output):
    assert is_prim_rou(q=q, d=d, val=i) == expected_output


def test_get_prim_rou_and_rou_inv_value_errors():
    with pytest.raises(ValueError):
        get_prim_rou_and_rou_inv(q=small_non_prime_for_testing, d=small_d_for_testing)
    with pytest.raises(ValueError):
        get_prim_rou_and_rou_inv(q=small_non_ntt_prime_for_testing, d=small_d_for_testing)


GET_PRIM_ROU_AND_ROU_INV_CASES = [
    (257, 64, (9, 200)),
    (8380417, 256, (1753, 731434)),
    (12289, 2, (1479, 10810)),
    (12289, 4, (4043, 5146)),
    (12289, 8, (722, 6553)),
    (12289, 16, (1212, 2545)),
    (12289, 32, (563, 5828)),
    (12289, 64, (81, 11227)),
    (12289, 128, (9, 2731)),
    (12289, 256, (3, 8193)),
    (12289, 512, (49, 1254)),
    (12289, 1024, (7, 8778)),
    (12289, 2048, (41, 4496)),
]


@pytest.mark.parametrize("q,d,expected_output", GET_PRIM_ROU_AND_ROU_INV_CASES)
def test_get_prim_rou_and_rou_inv(q, d, expected_output):
    x, y = get_prim_rou_and_rou_inv(q=q, d=d)
    assert x, y == expected_output
    assert x != 0
    assert y != 0
    assert (x * y) % q == 1
    assert all(x ** k % q != 0 for k in range(2, 2*d))


IS_BITSTRING_CASES = [
    ('0101100101', True),
    ('1010011010', True),
    (8675309, False),
    ('hello world', False)
]


@pytest.mark.parametrize("x,expected_output", IS_BITSTRING_CASES)
def test_is_bitstring(x, expected_output):
    assert is_bitstring(val=x) == expected_output


BIT_REV_CASES = [
    (2, 0, 0),
    (2, 1, 2),
    (2, 2, 1),
    (2, 3, 3),
    (3, 0, 0),
    (3, 1, 4),
    (3, 2, 2),
    (3, 3, 6),
    (3, 4, 1),
    (3, 5, 5),
    (3, 6, 3),
    (3, 7, 7),
    (4, 0, 0),
    (4, 1, 8),
    (4, 2, 4),
    (4, 3, 12),
    (4, 4, 2),
    (4, 5, 10),
    (4, 6, 6),
    (4, 7, 14),
    (4, 8, 1),
    (4, 9, 9),
    (4, 10, 5),
    (4, 11, 13),
    (4, 12, 3),
    (4, 13, 11),
    (4, 14, 7),
    (4, 15, 15),
]


@pytest.mark.parametrize("n,val,expected_output", BIT_REV_CASES)
def test_bit_rev(n, val, expected_output):
    assert bit_rev(num_bits=n, val=val) == expected_output


BIT_REV_CP_CASES = [
    ([0, 1], [0, 1]),
    ([0, 1, 2, 3], [0, 2, 1, 3]),
    ([0, 1, 2, 3, 4, 5, 6, 7], [0, 4, 2, 6, 1, 5, 3, 7]),
    ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15])
]


@pytest.mark.parametrize("x,expected_output", BIT_REV_CP_CASES)
def test_bit_rev_cp(x, expected_output):
    assert bit_rev_cp(x, ceil(log2(len(x)))) == expected_output


CENT_CASES = [
    (17, 8, 5, 0, 0),  # (q: int, halfmod: int, logmod: int, val: int)
    (17, 8, 5, 1, 1),
    (17, 8, 5, 2, 2),
    (17, 8, 5, 3, 3),
    (17, 8, 5, 4, 4),
    (17, 8, 5, 5, 5),
    (17, 8, 5, 6, 6),
    (17, 8, 5, 7, 7),
    (17, 8, 5, 8, 8),
    (17, 8, 5, 9, -8),
    (17, 8, 5, 10, -7),
    (17, 8, 5, 11, -6),
    (17, 8, 5, 12, -5),
    (17, 8, 5, 13, -4),
    (17, 8, 5, 14, -3),
    (17, 8, 5, 15, -2),
    (17, 8, 5, 16, -1),
]


@pytest.mark.parametrize("q,halfmod,logmod,val,expected_output", CENT_CASES)
def test_cent(q, halfmod, logmod, val, expected_output):
    assert cent(q=q, halfmod=halfmod, logmod=logmod, val=val) == expected_output


ZETAS_AND_INVS_CASES = [
    (17, 8, 8, 5, 16, 4, ([-1, -4, -8, 3], [-1, 4, 2, 6]))
]


@pytest.mark.parametrize("q,d,halfmod,logmod,n,lgn,expected_output", ZETAS_AND_INVS_CASES)
def test_make_zetas_and_invs(q, d, halfmod, logmod, n, lgn, expected_output):
    assert make_zetas_and_invs(q=q, d=d, halfmod=halfmod, logmod=logmod, n=n, lgn=lgn) == expected_output


ZETAS = [
    (17, 8, 8, 5, 16, 4, 3, 6, ([-1, -4, -8, 3], [-1, 4, 2, 6])),  # (q, d, halfmod, logmod, n, lgn, expected_output)
    (17, 8, 8, 5, 16, 4, 1, 1, ([1, 1, 1, 1], [1, 1, 1, 1])),
    (17, 8, 8, 5, 16, 4, 1, 2, ([1, 1, 1, 1], [1, -1, 4, 2])),
    (17, 8, 8, 5, 16, 4, 1, 3, ([1, 1, 1, 1], [-1, -4, -8, 3])),
]


@pytest.mark.parametrize("q,d,halfmod,logmod,n,lgn,rou,rou_inv,expected_output", ZETAS)
def test_make_zetas(mocker, q, d, halfmod, logmod, n, lgn, rou, rou_inv, expected_output):
    mocker.patch('lattice_algebra.main.get_prim_rou_and_rou_inv', return_value=(rou, rou_inv))
    assert make_zetas_and_invs(q=q, d=d, halfmod=halfmod, logmod=logmod, n=n, lgn=lgn) == expected_output


# We only test the NTT and inverse of constant polynomials here; more thorough tests are advisable
NTT_CASES = [
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], False, 8, 5, 16, 4, [1] + [0] * 15, [1] * 16),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], False, 8, 5, 16, 4, [2] + [0] * 15, [2] * 16),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], False, 8, 5, 16, 4, [3] + [0] * 15, [3] * 16),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], False, 8, 5, 16, 4, [4] + [0] * 15, [4] * 16),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], False, 8, 5, 16, 4, [5] + [0] * 15, [5] * 16),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], False, 8, 5, 16, 4, [6] + [0] * 15, [6] * 16),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], False, 8, 5, 16, 4, [7] + [0] * 15, [7] * 16),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], False, 8, 5, 16, 4, [8] + [0] * 15, [8] * 16),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], False, 8, 5, 16, 4, [-8] + [0] * 15, [-8] * 16),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], False, 8, 5, 16, 4, [-7] + [0] * 15, [-7] * 16),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], False, 8, 5, 16, 4, [-6] + [0] * 15, [-6] * 16),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], False, 8, 5, 16, 4, [-5] + [0] * 15, [-5] * 16),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], False, 8, 5, 16, 4, [-4] + [0] * 15, [-4] * 16),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], False, 8, 5, 16, 4, [-3] + [0] * 15, [-3] * 16),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], False, 8, 5, 16, 4, [-2] + [0] * 15, [-2] * 16),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], False, 8, 5, 16, 4, [-1] + [0] * 15, [-1] * 16),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], True, 8, 5, 16, 4, [1] * 16, [1] + [0] * 15),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], True, 8, 5, 16, 4, [2] * 16, [2] + [0] * 15),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], True, 8, 5, 16, 4, [3] * 16, [3] + [0] * 15),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], True, 8, 5, 16, 4, [4] * 16, [4] + [0] * 15),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], True, 8, 5, 16, 4, [5] * 16, [5] + [0] * 15),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], True, 8, 5, 16, 4, [6] * 16, [6] + [0] * 15),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], True, 8, 5, 16, 4, [7] * 16, [7] + [0] * 15),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], True, 8, 5, 16, 4, [8] * 16, [8] + [0] * 15),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], True, 8, 5, 16, 4, [-8] * 16, [-8] + [0] * 15),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], True, 8, 5, 16, 4, [-7] * 16, [-7] + [0] * 15),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], True, 8, 5, 16, 4, [-6] * 16, [-6] + [0] * 15),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], True, 8, 5, 16, 4, [-5] * 16, [-5] + [0] * 15),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], True, 8, 5, 16, 4, [-4] * 16, [-4] + [0] * 15),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], True, 8, 5, 16, 4, [-3] * 16, [-3] + [0] * 15),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], True, 8, 5, 16, 4, [-2] * 16, [-2] + [0] * 15),
    (17, [-1, -4, -8, 3], [-1, 4, 2, 6], True, 8, 5, 16, 4, [-1] * 16, [-1] + [0] * 15),
]


@pytest.mark.parametrize("q,zetas,zetas_inv,inv_flag,halfmod,logmod,n,lgn,val,expected_output", NTT_CASES)
def test_ntt(q, zetas, zetas_inv, inv_flag, halfmod, logmod, n, lgn, val, expected_output):
    assert expected_output == ntt(
        q=q, zetas=zetas, zetas_inv=zetas_inv, inv_flag=inv_flag, halfmod=halfmod, logmod=logmod, n=n, lgn=lgn, val=val
    )


small_bd_for_testing: int = 2
small_wt_for_testing: int = 2
small_dist_pars: dict[str, int] = {'bd': small_bd_for_testing, 'wt': small_wt_for_testing}
bits_to_decode_for_testing: int = ceil(log2(small_bd_for_testing)) + 1 + secpar4testing
bits_to_indices_for_testing: int = ceil(log2(degree_for_testing))
bits_to_indices_for_testing += (small_wt_for_testing - 1) * (ceil(log2(degree_for_testing)) + secpar4testing)

DECODE2COEF_CASES = [
    (
        secpar4testing,
        UNIFORM_INFINITY_WEIGHT,
        lp_for_testing,
        small_dist_pars,
        '0' + bin(i)[2:].zfill(bits_to_decode_for_testing - 1),
        - (i % small_bd_for_testing) - 1
    ) for i in range(2**7)
] + [
    (
        secpar4testing,
        UNIFORM_INFINITY_WEIGHT,
        lp_for_testing,
        small_dist_pars,
        '1' + bin(i)[2:].zfill(bits_to_decode_for_testing - 1),
        (i % small_bd_for_testing) + 1
    ) for i in range(2**7)
]


@pytest.mark.parametrize("secpar, dist, lp, dist_pars, val, expected_output", DECODE2COEF_CASES)
def test_decode2coef(secpar, dist, lp, dist_pars, val, expected_output):
    k = ceil(log2(dist_pars['bd'])) + 1 + secpar
    assert expected_output == decode2coef(
        secpar=secpar, lp=lp, distribution=dist, dist_pars=dist_pars, val=val, bits_to_decode=k
    )


DECODE2COEFS_CASES = [
    (
        secpar4testing,
        UNIFORM_INFINITY_WEIGHT,
        lp_for_testing,
        small_dist_pars,
        bin(randbits(2 * bits_to_decode_for_testing * small_dist_pars['wt'])),
        [(1+(i % (lp_for_testing.modulus//2)))*sign_i, (1+(j % (lp_for_testing.modulus//2)))*sign_j],
        [(1+(i % (lp_for_testing.modulus//2)))*sign_i, (1+(j % (lp_for_testing.modulus//2)))*sign_j],
    ) for i in range(2**7) for j in range(2**3) for sign_i in [-1, 1] for sign_j in [-1, 1]
]


@pytest.mark.parametrize("secpar,dist,lp,dist_pars,val,responses,expected_output", DECODE2COEFS_CASES)
def test_decode2coefs(mocker, secpar, dist, lp, dist_pars, val, responses, expected_output):
    assert responses == expected_output

    mocker.patch('lattice_algebra.main.decode2coef', side_effect=responses)
    assert dist == UNIFORM_INFINITY_WEIGHT
    k = ceil(log2(dist_pars['bd'])) + 1 + secpar
    observed_output = decode2coefs(
        secpar=secpar, lp=lp, distribution=dist, dist_pars=dist_pars, val=val,
        num_coefs=dist_pars['wt'], bits_to_decode=k
    )
    assert observed_output == expected_output


def int2bin(x: int, n: int):
    return bin(x)[2:].zfill(n)


def logdeg(d: int):
    return ceil(log2(d))


DECODE2INDICES_CASES = [
    (
        secpar4testing,
        UNIFORM_INFINITY_WEIGHT,
        lp_for_testing,
        small_dist_pars,
        int2bin(x=a, n=logdeg(lp_for_testing.degree)) + int2bin(x=b, n=logdeg(lp_for_testing.degree) + secpar4testing),
        small_wt_for_testing,
        [a, b] if b < a else [a, b + 1],
    ) for a in range(2 ** 5) for b in range(2 ** 5)
]


@pytest.mark.parametrize("secpar,dist,lp,dist_pars,val,num_coefs,expected_output", DECODE2INDICES_CASES)
def test_decode2indices(secpar, dist, lp, dist_pars, val, num_coefs, expected_output):
    observed_output = decode2indices(
        secpar=secpar, lp=lp, num_coefs=dist_pars['wt'], val=val, bits_to_indices=bits_to_indices_for_testing
    )
    assert observed_output == expected_output


DECODE2POLYCOEFS_CASES = [
    (
        secpar4testing,
        lp_for_testing,
        UNIFORM_INFINITY_WEIGHT,
        {'wt': small_wt_for_testing, 'bd': small_bd_for_testing},
        bin(
            randbits(small_wt_for_testing * bits_to_decode_for_testing + bits_to_indices_for_testing)
        )[2:].zfill(
            small_wt_for_testing * bits_to_decode_for_testing + bits_to_indices_for_testing
        ),
        small_wt_for_testing,
        bits_to_indices_for_testing,
        bits_to_decode_for_testing,
        [(i + j) % lp_for_testing.degree for j in range(small_wt_for_testing)],
        [2**j % lp_for_testing.modulus for j in range(small_wt_for_testing)],
        {(i + j) % lp_for_testing.degree: 2**j % lp_for_testing.modulus for j in range(small_wt_for_testing)}
    ) for i in range(2**10)
]


@pytest.mark.parametrize(
    "secpar,lp,dist,dist_pars,val,num_coefs,bits_to_indices," +
    "bits_to_decode,expected_indices,expected_coefs,expected_output",
    DECODE2POLYCOEFS_CASES
)
def test_decode2polycoefs(
        mocker, secpar, lp, dist, dist_pars, val, num_coefs, bits_to_indices, bits_to_decode,
        expected_indices, expected_coefs, expected_output
):
    mocker.patch("lattice_algebra.main.decode2indices", return_value=expected_indices)
    mocker.patch("lattice_algebra.main.decode2coefs", return_value=expected_coefs)
    assert expected_output == decode2polycoefs(
        secpar=secpar, lp=lp, distribution=dist, dist_pars=dist_pars, val=val,
        num_coefs=num_coefs, bits_to_indices=bits_to_indices, bits_to_decode=bits_to_decode
    )


exp_bits_per_poly_for_testing: int = int(log2(lp_for_testing.degree))
exp_bits_per_poly_for_testing += (small_dist_pars['wt'] - 1) * (int(log2(lp_for_testing.degree)) + secpar4testing)
exp_bits_per_poly_for_testing += small_dist_pars['wt']
exp_bits_per_poly_for_testing += small_dist_pars['wt'] * (ceil(log2(small_dist_pars['bd'])) + secpar4testing)
exp_bytes_per_poly_for_testing = ceil(exp_bits_per_poly_for_testing / 8)
GET_GEN_BYTES_PER_POLY_CASES = [
    (
        secpar4testing,
        lp_for_testing,
        UNIFORM_INFINITY_WEIGHT,
        small_dist_pars,
        small_wt_for_testing,
        bits_to_indices_for_testing,
        bits_to_decode_for_testing,
        exp_bytes_per_poly_for_testing
    )
]


@pytest.mark.parametrize(
    "secpar,lp,dist,dist_pars,num_coefs, bits_to_indices, bits_to_decode,expected_output",
    GET_GEN_BYTES_PER_POLY_CASES
)
def test_get_gen_bits(secpar, lp, dist, dist_pars, num_coefs, bits_to_indices, bits_to_decode, expected_output):
    observed_output = get_gen_bytes_per_poly(
        secpar=secpar, lp=lp, distribution=dist, dist_pars=dist_pars, num_coefs=num_coefs,
        bits_to_indices=bits_to_indices, bits_to_decode=bits_to_decode
    )
    assert observed_output == expected_output


# TODO: Modify the following tests to be properly parameterized, mocked, and fixtured
@pytest.fixture
def one() -> Polynomial:
    return Polynomial(lp=lp_for_testing, coefs={0: 1})


@pytest.fixture
def some_ran_lin_polys() -> List[Tuple[int, int, Polynomial]]:
    result = []
    for i in range(2 * sample_size_for_random_tests):
        a = (2 * randbits(1) - 1) * (randbelow(lp_for_testing.halfmod) + 1)
        b = (2 * randbits(1) - 1) * (randbelow(lp_for_testing.halfmod) + 1)
        result += [(a, b, Polynomial(lp=lp_for_testing, coefs={0: a, 1: b}))]
    return result


# @pytest.mark.skip
def test_polynomial_init(one, some_ran_lin_polys):
    lp: LatticeParameters = lp_for_testing

    # First, let's mess with the identity polynomial
    assert one.ntt_representation == [1 for _ in range(n_for_testing)]

    # Now let's make some random linear polynomials
    for next_tuple in some_ran_lin_polys:
        a, b, f = next_tuple
        assert -lp.halfmod <= a <= lp.halfmod
        assert -lp.halfmod <= b <= lp.halfmod
        assert isinstance(f, Polynomial)
        assert f.ntt_representation == [
            cent(q=lp.modulus, halfmod=lp.halfmod, logmod=lp.logmod, val=a + b * lp.rou ** k) for k in range(lp.n)
        ]


# @pytest.mark.skip
def test_polynomial_eq(one, some_ran_lin_polys):
    lp: LatticeParameters = lp_for_testing

    # First, let's make two identity polynomials and check they are equal.
    another_one = Polynomial(lp=lp, coefs={0: 1})
    assert one == another_one

    # Now check that if we change a single coefficient, the result changes
    not_one = Polynomial(lp=lp, coefs={0: -1})
    assert one != not_one

    # Now let's do the same with some random linear polynomials
    for next_tuple in some_ran_lin_polys:
        a, b, next_poly = next_tuple
        another_poly = Polynomial(lp=lp, coefs={0: a, 1: b})
        assert next_poly == another_poly

        # Now check that if we change a single coefficient, the result changes
        not_another_poly = Polynomial(lp=lp, coefs={0: a, 1: -b})
        assert next_poly != not_another_poly


@pytest.fixture
def two() -> Polynomial:
    return Polynomial(lp=lp_for_testing, coefs={0: 2})


@pytest.fixture
def pairs_ran_lin_poly(some_ran_lin_polys) -> List[Tuple[Tuple[int, int, Polynomial], Tuple[int, int, Polynomial]]]:
    result = []
    for i in range(0, sample_size_for_random_tests, 2):
        result += [(some_ran_lin_polys[i], some_ran_lin_polys[i + 1])]
    return result


@pytest.fixture
def pairs_of_random_polys_and_their_sums(pairs_ran_lin_poly) -> List[
    Tuple[
        Tuple[int, int, Polynomial],
        Tuple[int, int, Polynomial],
        Tuple[int, int, Polynomial, int, int],
        Tuple[dict, Polynomial, int, int]
    ]
]:
    result = []
    for next_pair in pairs_ran_lin_poly:
        next_f, next_g = next_pair
        a_f, b_f, f = next_f
        a_g, b_g, g = next_g
        observed_h = f + g
        obs_h_coefs, obs_h_norm, obs_h_wt = observed_h.get_coef_rep(const_time_flag=False)
        a_h = cent(
            q=lp_for_testing.modulus, val=a_f + a_g, halfmod=lp_for_testing.halfmod, logmod=lp_for_testing.logmod
        )
        b_h = cent(
            q=lp_for_testing.modulus, val=b_f + b_g, halfmod=lp_for_testing.halfmod, logmod=lp_for_testing.logmod
        )
        expected_h_coefs = {}
        if a_h != 0:
            expected_h_coefs[0] = a_h
        if b_h != 0:
            expected_h_coefs[1] = b_h
        expected_h = Polynomial(lp=lp_for_testing, coefs=expected_h_coefs)
        expected_h_norm = max(abs(a_h), abs(b_h))
        expected_h_wt = len(expected_h_coefs)
        result += [
            (
                next_f,
                next_g,
                (a_h, b_h, expected_h, expected_h_norm, expected_h_wt),
                (obs_h_coefs, observed_h, obs_h_norm, obs_h_wt)
            )
        ]
    return result


# @pytest.mark.skip
def test_polynomial_add(one, two, pairs_of_random_polys_and_their_sums):
    lp = lp_for_testing
    # First, let's make an identity polynomials and add it to itself
    assert one + one == two

    # Now let's do some addition with some random linear polynomials (AND the unity)
    for next_item in pairs_of_random_polys_and_their_sums:
        f_dat, g_dat, expected_h_dat, observed_h_dat = next_item
        a_f, b_f, f = f_dat
        a_g, b_g, g = g_dat
        a_h, b_h, exp_h, exp_h_norm, exp_h_wt = expected_h_dat
        obs_h_coefs, obs_h, obs_h_norm, obs_h_wt = observed_h_dat
        assert f + g == exp_h == obs_h
        assert len(obs_h_coefs) == 2
        assert 0 in obs_h_coefs
        assert 1 in obs_h_coefs
        assert cent(q=lp.modulus, halfmod=lp.halfmod, logmod=lp.logmod, val=obs_h_coefs[0] - a_h) == 0
        assert cent(q=lp.modulus, halfmod=lp.halfmod, logmod=lp.logmod, val=obs_h_coefs[1] - b_h) == 0
        assert cent(q=lp.modulus, halfmod=lp.halfmod, logmod=lp.logmod, val=a_f + a_g - a_h) == 0
        assert cent(q=lp.modulus, halfmod=lp.halfmod, logmod=lp.logmod, val=b_f + b_g - b_h) == 0


@pytest.fixture
def pairs_of_random_polys_and_their_diffs(pairs_ran_lin_poly) -> \
        List[
            Tuple[
                Tuple[int, int, Polynomial],
                Tuple[int, int, Polynomial],
                Tuple[int, int, Polynomial, int, int],
                Tuple[Dict[int, int], Polynomial, int, int]]
        ]:
    result = []
    for next_pair in pairs_ran_lin_poly:
        next_f, next_g = next_pair
        a_f, b_f, f = next_f
        a_g, b_g, g = next_g
        a_h = cent(
            q=lp_for_testing.modulus, val=a_f - a_g, halfmod=lp_for_testing.halfmod, logmod=lp_for_testing.logmod
        )
        b_h = cent(
            q=lp_for_testing.modulus, val=b_f - b_g, halfmod=lp_for_testing.halfmod, logmod=lp_for_testing.logmod
        )
        expected_h_coefs: dict = {}
        if a_h != 0:
            expected_h_coefs[0] = a_h
        if b_h != 0:
            expected_h_coefs[1] = b_h
        expected_h_norm: int = max(abs(expected_h_coefs[i]) for i in expected_h_coefs)
        expected_h_wt: int = len(expected_h_coefs)

        observed_h = f - g
        obs_h_coefs, obs_h_norm, obs_h_wt = observed_h.get_coef_rep(const_time_flag=False)
        expected_h = Polynomial(lp=lp_for_testing, coefs=expected_h_coefs)
        result += [
            (
                next_f,
                next_g,
                (a_h, b_h, expected_h, expected_h_norm, expected_h_wt),
                (obs_h_coefs, observed_h, obs_h_norm, obs_h_wt)
            )
        ]
    return result


# @pytest.mark.skip
def test_polynomial_sub(pairs_of_random_polys_and_their_diffs):
    lp: LatticeParameters = lp_for_testing
    # Now let's do some addition with some random linear polynomials (AND the unity)
    for next_item in pairs_of_random_polys_and_their_diffs:
        f_dat, g_dat, expected_h_dat, observed_h_dat = next_item
        a_f, b_f, f = f_dat
        a_g, b_g, g = g_dat
        a_h, b_h, exp_h, exp_h_norm, exp_h_wt = expected_h_dat
        obs_h_coefs, obs_h, obs_h_norm, obs_h_wt = observed_h_dat
        assert f - g == exp_h == obs_h
        assert len(obs_h_coefs) == 2
        assert 0 in obs_h_coefs
        assert 1 in obs_h_coefs
        assert cent(q=lp.modulus, halfmod=lp.halfmod, logmod=lp.logmod, val=obs_h_coefs[0] - a_h) == 0
        assert cent(q=lp.modulus, halfmod=lp.halfmod, logmod=lp.logmod, val=obs_h_coefs[1] - b_h) == 0
        assert cent(q=lp.modulus, halfmod=lp.halfmod, logmod=lp.logmod, val=a_f - a_g - a_h) == 0
        assert cent(q=lp.modulus, halfmod=lp.halfmod, logmod=lp.logmod, val=b_f - b_g - b_h) == 0


@pytest.fixture
def pairs_of_random_polys_and_their_products(pairs_ran_lin_poly) -> \
        List[
            Tuple[
                Tuple[int, int, Polynomial],
                Tuple[int, int, Polynomial],
                Tuple[int, int, int, Polynomial, int, int],
                Tuple[Dict[int, int], Polynomial, int, int]
            ]
        ]:
    result = []
    for next_pair in pairs_ran_lin_poly:
        next_f, next_g = next_pair
        a_f, b_f, f = next_f
        a_g, b_g, g = next_g
        a_h = cent(
            q=lp_for_testing.modulus, halfmod=lp_for_testing.halfmod, logmod=lp_for_testing.logmod, val=a_f * a_g
        )
        b_h = cent(
            q=lp_for_testing.modulus,
            halfmod=lp_for_testing.halfmod,
            logmod=lp_for_testing.logmod,
            val=a_f * b_g + b_f * a_g
        )
        c_h = cent(
            q=lp_for_testing.modulus, halfmod=lp_for_testing.halfmod, logmod=lp_for_testing.logmod, val=b_f * b_g
        )
        exp_h_coefs = {}
        if a_h != 0:
            exp_h_coefs[0] = a_h
        if b_h != 0:
            exp_h_coefs[1] = b_h
        if c_h != 0:
            exp_h_coefs[2] = c_h
        exp_h_norm = max(abs(a_h), abs(b_h), abs(c_h))
        exp_h_wt = len(exp_h_coefs)
        observed_h = f * g
        obs_h_coefs, obs_h_norm, obs_h_wt = observed_h.get_coef_rep(const_time_flag=False)
        expected_h = Polynomial(lp=lp_for_testing, coefs=exp_h_coefs)
        result += [
            (
                next_f,
                next_g,
                (a_f * a_g, a_f * b_g + a_g * b_f, b_f * b_g, expected_h, exp_h_norm, exp_h_wt),
                (obs_h_coefs, observed_h, obs_h_norm, obs_h_wt)
            )
        ]
    return result


# @pytest.mark.skip
def test_polynomial_mul_small(one, pairs_of_random_polys_and_their_products):
    lp = lp_for_testing
    # First, let's make an identity polynomials and add it to itself
    assert one * one == one

    # Now let's do some addition with some random linear polynomials (AND the unity)
    for next_item in pairs_of_random_polys_and_their_products:
        f_dat, g_dat, expected_h_dat, observed_h_dat = next_item
        a_f, b_f, f = f_dat
        a_g, b_g, g = g_dat
        a_h, b_h, c_h, exp_h, exp_h_norm, exp_h_wt = expected_h_dat
        obs_h_coefs, obs_h, obs_h_norm, obs_h_wt = observed_h_dat
        assert one * f == f
        assert f * one == f
        assert one * g == g
        assert g * one == g
        assert f * g == exp_h == obs_h
        assert len(obs_h_coefs) == 3
        assert 0 in obs_h_coefs
        assert 1 in obs_h_coefs
        assert 2 in obs_h_coefs
        assert cent(q=lp.modulus, halfmod=lp.halfmod, logmod=lp.logmod, val=obs_h_coefs[0] - a_h) == 0
        assert cent(q=lp.modulus, halfmod=lp.halfmod, logmod=lp.logmod, val=obs_h_coefs[1] - b_h) == 0
        assert cent(q=lp.modulus, halfmod=lp.halfmod, logmod=lp.logmod, val=obs_h_coefs[2] - c_h) == 0
        assert cent(q=lp.modulus, halfmod=lp.halfmod, logmod=lp.logmod, val=a_f * a_g - a_h) == 0
        assert cent(q=lp.modulus, halfmod=lp.halfmod, logmod=lp.logmod, val=a_f * b_g + b_f * a_g - b_h) == 0
        assert cent(q=lp.modulus, halfmod=lp.halfmod, logmod=lp.logmod, val=b_f * b_g - c_h) == 0


# @pytest.mark.skip
def test_polynomial_repr(some_ran_lin_polys):
    for next_item in some_ran_lin_polys:
        a, b, f = next_item
        coef_rep, norm, wt = f.get_coef_rep(const_time_flag=False)
        assert norm == max(abs(a), abs(b))
        if a != 0 and b != 0:
            assert wt == 2
        elif a != 0 or b != 0:
            assert wt == 1
        else:
            assert wt == 0

        sorted_keys = sorted(list(coef_rep.keys()))
        sorted_coefs = [(i, coef_rep[i]) for i in sorted_keys]
        assert str(f) == str((sorted_coefs, norm, wt))


# @pytest.mark.skip
def test_polynomial_reset_vals(one):
    x = deepcopy(one)
    x._reset_vals(coefs={0: 2})
    assert x.ntt_representation == [2 for _ in range(lp_for_testing.n)]
    x._reset_vals(coefs={0: 1})
    assert x == one


# @pytest.mark.skip
def test_polynomial_get_coefs(one):
    x = Polynomial(lp=lp_for_testing, coefs={1: 1})
    f = one + x
    assert f.get_coef_rep(const_time_flag=False) == ({0: 1, 1: 1}, 1, 2)


# @pytest.mark.skip
def test_polynomial_norm_and_weight(some_ran_lin_polys):
    for next_item in some_ran_lin_polys:
        a, b, f = next_item
        f_coefs, n, w = f.get_coef_rep(const_time_flag=False)
        assert n == max(abs(a), abs(b))
        if a != 0 and b != 0:
            assert w == 2
        elif (a == 0 and b != 0) or (a != 0 and b == 0):
            assert w == 1
        else:
            assert w == 0


# @pytest.mark.skip
def test_rand_poly():
    f = random_polynomial(
        secpar=secpar4testing, lp=lp_for_testing, distribution=UNIFORM_INFINITY_WEIGHT,
        dist_pars=small_dist_pars, num_coefs=small_dist_pars['wt'], bits_to_indices=bits_to_indices_for_testing,
        bits_to_decode=bits_to_decode_for_testing
    )
    assert isinstance(f, Polynomial)
    f_coefs, n, w = f.get_coef_rep(const_time_flag=False)
    assert n <= small_dist_pars['bd'] and w <= small_dist_pars['bd']
    assert max(abs(f_coefs[i]) for i in f_coefs) <= n
    assert len(f_coefs) <= w


@pytest.fixture
def some_random_polys_for_a_vector() -> List[Polynomial]:
    lp: LatticeParameters = lp_for_testing
    return [random_polynomial(
        secpar=secpar4testing, lp=lp, distribution=UNIFORM_INFINITY_WEIGHT, dist_pars=small_dist_pars,
        num_coefs=small_dist_pars['wt'], bits_to_indices=bits_to_indices_for_testing,
        bits_to_decode=bits_to_decode_for_testing
    ) for _ in range(lp.length)]


# @pytest.mark.skip
def test_polynomial_vector_init(some_random_polys_for_a_vector):
    lp: LatticeParameters = lp_for_testing
    assert PolynomialVector(lp=lp, entries=some_random_polys_for_a_vector)
    v = PolynomialVector(lp=lp, entries=some_random_polys_for_a_vector)
    tmp = v.get_coef_rep(const_time_flag=False)
    assert max(i[1] for i in tmp) <= lp.halfmod // 2
    assert max(i[2] for i in tmp) <= lp.degree
    assert random_polynomialvector(
        secpar=secpar4testing, lp=lp, distribution=UNIFORM_INFINITY_WEIGHT, dist_pars=small_dist_pars,
        num_coefs=small_dist_pars['wt'], bits_to_indices=bits_to_indices_for_testing,
        bits_to_decode=bits_to_decode_for_testing
    )


@pytest.fixture
def some_random_polynomialvector(some_random_polys_for_a_vector) -> PolynomialVector:
    return PolynomialVector(lp=lp_for_testing, entries=some_random_polys_for_a_vector)


@pytest.fixture
def some_random_polynomialvectors(some_random_polys_for_a_vector) -> List[PolynomialVector]:
    return [PolynomialVector(lp=lp_for_testing, entries=some_random_polys_for_a_vector) for _ in
            range(sample_size_for_random_tests)]


# @pytest.mark.skip
def test_polynomial_vector_eq(some_random_polys_for_a_vector, some_random_polynomialvector):
    lp: LatticeParameters = lp_for_testing
    v = PolynomialVector(lp=lp, entries=deepcopy(some_random_polys_for_a_vector))
    assert v == some_random_polynomialvector


@pytest.fixture
def some_random_polynomialvector_pairs_sums() -> List[Tuple[PolynomialVector, PolynomialVector, PolynomialVector]]:
    result = []
    while len(result) < sample_size_for_random_tests:
        f = random_polynomialvector(
            secpar=secpar4testing, lp=lp_for_testing, distribution=UNIFORM_INFINITY_WEIGHT, dist_pars=small_dist_pars,
            num_coefs=small_dist_pars['wt'], bits_to_indices=bits_to_indices_for_testing,
            bits_to_decode=bits_to_decode_for_testing
        )
        g = random_polynomialvector(
            secpar=secpar4testing, lp=lp_for_testing, distribution=UNIFORM_INFINITY_WEIGHT, dist_pars=small_dist_pars,
            num_coefs=small_dist_pars['wt'], bits_to_indices=bits_to_indices_for_testing,
            bits_to_decode=bits_to_decode_for_testing
        )
        h = f + g
        result += [(f, g, h)]
    return result


@pytest.fixture
def some_random_polynomialvector_pairs_diffs() -> List[Tuple[PolynomialVector, PolynomialVector, PolynomialVector]]:
    result = []
    while len(result) < sample_size_for_random_tests:
        f = random_polynomialvector(
            secpar=secpar4testing, lp=lp_for_testing, distribution=UNIFORM_INFINITY_WEIGHT, dist_pars=small_dist_pars,
            num_coefs=small_dist_pars['wt'], bits_to_indices=bits_to_indices_for_testing,
            bits_to_decode=bits_to_decode_for_testing
        )
        g = random_polynomialvector(
            secpar=secpar4testing, lp=lp_for_testing, distribution=UNIFORM_INFINITY_WEIGHT, dist_pars=small_dist_pars,
            num_coefs=small_dist_pars['wt'], bits_to_indices=bits_to_indices_for_testing,
            bits_to_decode=bits_to_decode_for_testing
        )
        h = f - g
        result += [(f, g, h)]
    return result


# @pytest.mark.skip
def test_polynomial_vector_add(some_random_polynomialvector_pairs_sums):
    # TODO: Rewrite
    lp: LatticeParameters = lp_for_testing
    for next_item in some_random_polynomialvector_pairs_sums:
        f, g, observed_h = next_item
        for i, val in enumerate(zip(f.entries, g.entries, observed_h.entries)):
            ff, gg, hh = val
            diff = [ff.ntt_representation[j] + gg.ntt_representation[j] for j in range(lp.degree)]
            diff = [diff[j] - hh.ntt_representation[j] for j in range(lp.degree)]
            assert all(
                cent(q=lp.modulus, halfmod=lp.halfmod, logmod=lp.logmod, val=diff[j]) == 0 for j in range(lp.degree)
            )
        observed_h_coefs_and_norms_and_wts = observed_h.get_coef_rep(const_time_flag=False)
        f_coefs_and_norms_and_wts = f.get_coef_rep(const_time_flag=False)
        g_coefs_and_norms_and_wts = g.get_coef_rep(const_time_flag=False)
        for i, val in enumerate(zip(
                f_coefs_and_norms_and_wts, g_coefs_and_norms_and_wts, observed_h_coefs_and_norms_and_wts
        )):
            f_dat, g_dat, obs_h_dat = val
            f_coefs, f_norm, f_wt = f_dat
            g_coefs, g_norm, g_wt = g_dat
            expected_h_coefs: Dict[int, int] = deepcopy(f_coefs)
            obs_h_coefs, obs_h_norm, obs_h_wt = obs_h_dat
            for j in g_coefs:
                if j in expected_h_coefs:
                    diff = expected_h_coefs[j] + g_coefs[j]
                    expected_h_coefs[j] = cent(
                        q=lp_for_testing.modulus,
                        val=diff, halfmod=lp_for_testing.halfmod,
                        logmod=lp_for_testing.logmod
                    )
                else:
                    expected_h_coefs[j] = g_coefs[j]
            expected_h_coefs = {i: expected_h_coefs[i] for i in expected_h_coefs if expected_h_coefs[i] != 0}
            expected_h_norm: int = max(abs(expected_h_coefs[i]) for i in expected_h_coefs)
            expected_h_wt: int = len(expected_h_coefs)
            assert expected_h_wt == obs_h_wt
            assert expected_h_norm == obs_h_norm
            assert sorted(list(expected_h_coefs.keys())) == sorted(list(obs_h_coefs.keys()))
            assert all(
                0 == cent(
                    q=lp_for_testing.modulus,
                    val=expected_h_coefs[i] - obs_h_coefs[i],
                    halfmod=lp_for_testing.halfmod,
                    logmod=lp_for_testing.logmod
                ) for i in expected_h_coefs
            )


# @pytest.mark.skip
def test_polynomial_vector_sub(some_random_polynomialvector_pairs_diffs):
    lp: LatticeParameters = lp_for_testing
    for next_item in some_random_polynomialvector_pairs_diffs:
        f, g, observed_h = next_item
        observed_h_coefs_and_norms_and_weights = [
            i.get_coef_rep(const_time_flag=False) for i in observed_h.entries
        ]
        for i in range(lp.length):
            f_coefs, f_norm, f_wt = f.entries[i].get_coef_rep(const_time_flag=False)
            g_coefs, g_norm, g_wt = g.entries[i].get_coef_rep(const_time_flag=False)
            obs_h_coefs, obs_h_norm, obs_h_wt = observed_h_coefs_and_norms_and_weights[i]
            exp_h_coefs = deepcopy(f_coefs)
            for j in g_coefs:
                if j in exp_h_coefs:
                    diff = exp_h_coefs[j] - g_coefs[j]
                    exp_h_coefs[j] = cent(
                        q=lp_for_testing.modulus,
                        val=diff, halfmod=lp_for_testing.halfmod,
                        logmod=lp_for_testing.logmod
                    )
                else:
                    exp_h_coefs[j] = -g_coefs[j]
            exp_h_coefs = {i: exp_h_coefs[i] for i in exp_h_coefs if exp_h_coefs[i] != 0}
            exp_h_norm = max(abs(exp_h_coefs[i]) for i in exp_h_coefs)
            exp_h_wt = len(exp_h_coefs)

            assert obs_h_norm == exp_h_norm
            assert obs_h_wt == exp_h_wt
            assert sorted(list(obs_h_coefs.keys())) == sorted(list(exp_h_coefs.keys()))
            for j in obs_h_coefs:
                diff = obs_h_coefs[j] - exp_h_coefs[j]
                assert 0 == cent(
                    q=lp_for_testing.modulus, val=diff, halfmod=lp_for_testing.halfmod, logmod=lp_for_testing.logmod
                )


# @pytest.mark.skip
def test_polynomial_vector_mul(one, some_random_polynomialvectors):
    # tests the dot product
    lp: LatticeParameters = lp_for_testing
    all_ones: PolynomialVector = PolynomialVector(lp=lp, entries=[deepcopy(one) for _ in range(lp.length)])
    for v in some_random_polynomialvectors:
        expected_sum: Polynomial = sum(x for x in v.entries)
        observed_sum: Polynomial = all_ones * v
        assert observed_sum == expected_sum
        observed_sum: Polynomial = v * all_ones
        assert observed_sum == expected_sum


@pytest.fixture
def some_random_linear_polynomialvectors() -> list:
    result = []
    while len(result) < sample_size_for_random_tests:
        next_result_entry = []
        entries = []
        while len(entries) < lp_for_testing.length:
            a: int = (2 * randbits(1) - 1) * (randbelow(lp_for_testing.halfmod) + 1)
            b: int = (2 * randbits(1) - 1) * (randbelow(lp_for_testing.halfmod) + 1)
            if a != 0 or b != 0:
                next_poly: Polynomial = Polynomial(lp=lp_for_testing, coefs={0: a, 1: b})
                next_result_entry += [(a, b, next_poly)]
                entries += [next_poly]
        next_polynomialvector = PolynomialVector(lp=lp_for_testing, entries=entries)
        next_result_entry += [next_polynomialvector]
        result += [next_result_entry]
    return result


@pytest.fixture
def expected_polynomialvector_rep(some_random_linear_polynomialvectors) -> List[str]:
    result = []
    for next_polynomialvector_data in some_random_linear_polynomialvectors:
        next_rep: str = '['
        the_polynomialvector = next_polynomialvector_data[-1]
        for i, val in enumerate(zip(the_polynomialvector.entries, next_polynomialvector_data[:-1])):
            # the_next_poly = val[0]
            next_tuple = val[1]
            a, b, also_the_next_poly = next_tuple
            the_coefs = {}
            if a != 0:
                the_coefs[0] = a
            if b != 0:
                the_coefs[1] = b
            the_norm = max(abs(a), abs(b))
            the_wt = len(the_coefs)
            # assert the_next_poly.get_coef_rep() == the_coefs, the_norm, the_wt
            # assert also_the_next_poly == the_next_poly
            the_coefs_sorted_keys = sorted(list(the_coefs.keys()))
            sorted_coefs = [(i, the_coefs[i]) for i in the_coefs_sorted_keys]
            next_rep += str((sorted_coefs, the_norm, the_wt)) + ', '
        next_rep = next_rep[:-2] + ']'
        result += [next_rep]
    return result


# @pytest.mark.skip
def test_polynomial_vector_repr(some_random_linear_polynomialvectors, expected_polynomialvector_rep):
    for next_pair in zip(some_random_linear_polynomialvectors, expected_polynomialvector_rep):
        next_random_linear_polynomialvector, next_expected_rep = next_pair
        assert str(next_random_linear_polynomialvector[-1]) == next_expected_rep


# @pytest.mark.skip
def test_polynomial_vector_pow():
    for k in range(sample_size_for_random_tests):
        v: Polynomial = random_polynomial(
            secpar=secpar4testing, lp=lp_for_testing, distribution=UNIFORM_INFINITY_WEIGHT, dist_pars=small_dist_pars,
            num_coefs=small_dist_pars['wt'], bits_to_indices=bits_to_indices_for_testing,
            bits_to_decode=bits_to_decode_for_testing
        )
        u: PolynomialVector = random_polynomialvector(
            secpar=secpar4testing, lp=lp_for_testing, distribution=UNIFORM_INFINITY_WEIGHT, dist_pars=small_dist_pars,
            num_coefs=small_dist_pars['wt'], bits_to_indices=bits_to_indices_for_testing,
            bits_to_decode=bits_to_decode_for_testing
        )
        expected_scaled_vector: PolynomialVector = deepcopy(u)
        expected_scaled_vector.entries = [i * v for i in expected_scaled_vector.entries]
        observed_scaled_vector = u ** v
        assert expected_scaled_vector == observed_scaled_vector


# @pytest.mark.skip
def test_polynomialvector_coefficient_representation_and_norm_and_weight():
    f: PolynomialVector = random_polynomialvector(
        secpar=secpar4testing, lp=lp_for_testing, distribution=UNIFORM_INFINITY_WEIGHT, dist_pars=small_dist_pars,
        num_coefs=small_dist_pars['wt'], bits_to_indices=bits_to_indices_for_testing,
        bits_to_decode=bits_to_decode_for_testing
    )
    assert isinstance(f, PolynomialVector)
    result = f.get_coef_rep(const_time_flag=False)
    for i in result:
        coef_rep, n, w = i
        assert n <= 7176
        assert w <= 384
        assert w == len(coef_rep)
        assert n == max(abs(coef_rep[i]) for i in coef_rep)


# @pytest.mark.skip
def test_random_polynomialvector():
    f: PolynomialVector = random_polynomialvector(
        secpar=secpar4testing, lp=lp_for_testing, distribution=UNIFORM_INFINITY_WEIGHT, dist_pars=small_dist_pars,
        num_coefs=small_dist_pars['wt'], bits_to_indices=bits_to_indices_for_testing,
        bits_to_decode=bits_to_decode_for_testing
    )
    assert isinstance(f, PolynomialVector)
    results = f.get_coef_rep(const_time_flag=False)
    for i in results:
        coef_rep, n, w = i
        assert n <= 7176
        assert w <= 384
        assert len(coef_rep) == w
        assert n == max(abs(coef_rep[i]) for i in coef_rep)


# @pytest.mark.skip
def test_decode_bitstring_to_coefficient():
    # TODO: Rewrite this
    dist_pars_tmp = deepcopy(small_dist_pars)
    for bound in range(1, lp_for_testing.modulus // 2, lp_for_testing.modulus // (2**5)):
        dist_pars_tmp['bd'] = bound
        bits_to_decode = ceil(log2(bound)) + 1 + secpar4testing
        for signum_bit in range(2):
            for magnitude_minus_one in range(bound):
                expected_result = (2*signum_bit - 1)*(1 + magnitude_minus_one)
                bitstring = str(signum_bit) + bin(magnitude_minus_one)[2:].zfill(bits_to_decode-1)
                observed_result = decode2coef(
                    secpar=secpar4testing, lp=lp_for_testing, val=bitstring, distribution=UNIFORM_INFINITY_WEIGHT,
                    dist_pars=dist_pars_tmp, bits_to_decode=bits_to_decode
                )
                assert expected_result == observed_result


# @pytest.mark.skip
def test_decode_bitstring_to_indices():
    lp: LatticeParameters = LatticeParameters(degree=8, modulus=small_q_for_testing, length=3)
    dist_pars: dict[str, int] = {'bd': 1, 'wt': 3}
    bits_to_indices_for_this_test: int = ceil(log2(lp.degree)) + 2*(ceil(log2(lp.degree)) + secpar4testing)
    # say we want indices 0, 3, 6.
    # first index is 0, so the first part of the bitstring is '000' (only need 3 bits for the first)
    # second index is 3, requires ceil(log2(8)) + secpar = 11 bits since we already picked an index.
    # but the remaining indices are [1, 2, 3, 4, 5, 6, 7]
    # to get 3 out of this, we need to access index 2!!!
    # we need an 11-bit integer == 2 mod 7. for fun, let's use 2 + 2**6 * 7 = 450 -> '00111000010`
    # third index is 6, requires ceil(log2(8)) + secpar = 11 bits since we already picked two indices
    # but the remaining indices are [1, 2, 4, 5, 6, 7]
    # to get 6 out of this, we need to access index 4
    # we need an 11-bit integer == 4 mod 6. for fun, let's use 4 + 2**5 * 6 = 196 -> '00011000100'
    # i.e. set our bitstring = '0000011100001000011000100' and we should get
    # [0, 3, 6].
    val = bin(0)[2:].zfill(ceil(log2(lp.degree)))
    val += bin(2 + 2**6 * 7)[2:].zfill(ceil(log2(lp.degree)) + secpar4testing)
    val += bin(4 + 2**5 * 6)[2:].zfill(ceil(log2(lp.degree)) + secpar4testing)
    expected_result = [0, 3, 6]
    observed_result = decode2indices(
        secpar=secpar4testing, lp=lp, num_coefs=dist_pars['wt'], val=val, bits_to_indices=bits_to_indices_for_this_test
    )
    assert expected_result == observed_result


# For coefficients: length ceil(log2(bound)) + 1 + secpar
# for signum_bit in range(2):
#     for magnitude_minus_one in range(bound):
#         expected_result = (2*signum_bit - 1)*(1 + magnitude_minus_one)
#         bitstring = str(signum_bit) + bin(magnitude_minus_one)[2:].zfill(bits_to_decode-1)


# @pytest.mark.skip
def test_decode_bitstring_to_polynomial_coefficients():
    lp: LatticeParameters = LatticeParameters(degree=8, modulus=small_q_for_testing, length=3)

    # Let's construct the bitstring that should give us {0: 1, 3: 1, 6: -1}.
    expected_coefs = [1, 1, -1]
    expected_indices = [0, 3, 6]
    expected_result = {idx: coef for idx, coef in zip(expected_indices, expected_coefs)}
    # expected coefficients will be [1, 1, -1]
    # expected indices will be [0, 3, 6]
    val_for_first_index = bin(0)[2:].zfill(ceil(log2(lp.degree)))
    val_for_second_index = bin(2 + 2 ** 6 * 7)[2:].zfill(ceil(log2(lp.degree)) + secpar4testing)
    val_for_third_index = bin(4 + 2 ** 5 * 6)[2:].zfill(ceil(log2(lp.degree)) + secpar4testing)
    val_for_indices = val_for_first_index + val_for_second_index + val_for_third_index
    val_for_first_coef = '1' + '0' * (bits_to_decode_for_testing - 1)
    val_for_second_coef = '1' + '0' * (bits_to_decode_for_testing - 1)
    val_for_third_coef = '0' + '0' * (bits_to_decode_for_testing - 1)
    val_for_coefs = val_for_first_coef + val_for_second_coef + val_for_third_coef

    val = val_for_indices + val_for_coefs

    wt_for_this_test: int = 3
    bits_to_indices: int = ceil(log2(lp.degree)) + (wt_for_this_test - 1) * (ceil(log2(lp.degree)) + secpar4testing)

    for bd in range(1, small_q_for_testing//(2**5)):
        dist_pars: dict[str, int] = {'bd': bd, 'wt': wt_for_this_test}
        observed_result = decode2polycoefs(
            secpar=secpar4testing, lp=lp, distribution=UNIFORM_INFINITY_WEIGHT, dist_pars=dist_pars, val=val,
            num_coefs=dist_pars['wt'], bits_to_decode=bits_to_decode_for_testing, bits_to_indices=bits_to_indices
        )
        assert expected_result == observed_result


# @pytest.mark.skip
def test_decode_bitstring_to_coefficients():
    # Thorough test
    for bound in range(2, modulus_for_testing // 2, modulus_for_testing // (2**5)):
        for weight in range(1, degree_for_testing, degree_for_testing // (2**5)):
            expected_result = list()
            while len(expected_result) < weight:
                next_int_sign = 2 * randbits(1) - 1
                next_int_mag = 1 + randbelow(bound)
                expected_result += [next_int_sign * next_int_mag]
            num_bits = weight * (1 + ceil(log2(bound)) + secpar4testing)
            vals = list()
            for i in expected_result:
                tmp = bin(int(i > 0))[2:] + bin(abs(i) - 1)[2:].zfill(num_bits // weight - 1)
                vals += [tmp]
            assert len(vals) == weight
            for i in vals:
                assert len(i) == num_bits // weight
            merged_vals = ''
            for i in vals:
                merged_vals += i
            bits_to_decode = ceil(log2(bound)) + 1 + secpar4testing
            observed_result = decode2coefs(
                secpar=secpar4testing, lp=lp_for_testing, distribution=UNIFORM_INFINITY_WEIGHT,
                dist_pars={'bd': bound, 'wt': weight}, val=merged_vals, num_coefs=weight,
                bits_to_decode=bits_to_decode
            )
            assert expected_result == observed_result
