"""
Tests the lattices package.

Todo list:
 - more and more useful fixtures will make testing faster and more efficient
 - more and more rigorous tests
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


@pytest.fixture
def small_q_is_prime() -> bool:
    return is_prime(small_q_for_testing)


@pytest.fixture
def small_non_prime_is_not_prime() -> bool:
    return not is_prime(small_non_prime_for_testing)


# @pytest.mark.skip
def test_is_prime(small_q_is_prime, small_non_prime_is_not_prime):
    assert small_q_is_prime and small_non_prime_is_not_prime


# @pytest.mark.skip
def test_is_pow_two():
    for i in range(1, 10):
        assert is_pow_two(2 ** i)
        for j in range(1, 2 ** i):
            assert not is_pow_two(2 ** i + j)
        assert is_pow_two(2 ** i + 2 ** i)


@pytest.fixture
def small_q_small_d_are_ntt_friendly() -> bool:
    return is_ntt_friendly_prime(q=small_q_for_testing, d=small_d_for_testing)


@pytest.fixture
def small_non_prime_small_d_is_not_ntt_friendly() -> bool:
    return not is_ntt_friendly_prime(q=small_non_prime_for_testing, d=small_d_for_testing)


@pytest.fixture
def small_non_ntt_prime_is_not_ntt_friendly() -> bool:
    return not is_ntt_friendly_prime(q=small_non_ntt_prime_for_testing, d=small_d_for_testing)


# @pytest.mark.skip
def test_is_ntt_friendly_prime(small_q_small_d_are_ntt_friendly, small_non_prime_small_d_is_not_ntt_friendly,
                               small_non_ntt_prime_is_not_ntt_friendly):
    assert small_q_small_d_are_ntt_friendly and small_non_prime_small_d_is_not_ntt_friendly and \
           small_non_ntt_prime_is_not_ntt_friendly


@pytest.fixture
def scan_for_prim_rou() -> List[int]:
    return [
        i for i in range(2, small_q_for_testing) if is_prim_rou(q=small_q_for_testing, d=small_d_for_testing, val=i)
    ]


# @pytest.mark.skip
def test_is_prim_rou(scan_for_prim_rou):
    assert scan_for_prim_rou == allowed_primitive_roots_of_unity_for_small_q_and_small_d


@pytest.fixture
def small_q_small_d_prim_rou() -> Tuple[int, int]:
    return get_prim_rou_and_rou_inv(q=small_q_for_testing, d=small_d_for_testing)


@pytest.fixture
def prim_rou_and_inv_for_testing() -> Tuple[int, int]:
    return get_prim_rou_and_rou_inv(q=modulus_for_testing, d=degree_for_testing)


# @pytest.mark.skip
def test_get_prim_rou_and_rou_inv(small_q_small_d_prim_rou):
    with pytest.raises(ValueError):
        get_prim_rou_and_rou_inv(q=small_non_prime_for_testing, d=small_d_for_testing)
    with pytest.raises(ValueError):
        get_prim_rou_and_rou_inv(q=small_non_ntt_prime_for_testing, d=small_d_for_testing)
    assert not isinstance(small_q_small_d_prim_rou, bool) and isinstance(small_q_small_d_prim_rou, tuple)
    assert isinstance(small_q_small_d_prim_rou[0], int) and isinstance(small_q_small_d_prim_rou[1], int)
    assert small_q_small_d_prim_rou[0] is not None and small_q_small_d_prim_rou[1] is not None
    assert small_q_small_d_prim_rou[0] == min(allowed_primitive_roots_of_unity_for_small_q_and_small_d)
    assert small_q_small_d_prim_rou[0] * small_q_small_d_prim_rou[1] % small_q_for_testing == 1


# @pytest.mark.skip
def test_is_bitstring():
    assert is_bitstring('100001000101111111101101')
    assert not is_bitstring('hello world')


@pytest.fixture
def three_bit_rev() -> List[int]:
    return [bit_rev(3, i) for i in range(8)]


@pytest.fixture
def expected_three_bit_rev() -> List[int]:
    return [0, 4, 2, 6, 1, 5, 3, 7]


# @pytest.mark.skip
def test_bit_rev(three_bit_rev, expected_three_bit_rev):
    assert three_bit_rev == expected_three_bit_rev


@pytest.fixture
def three_bit_rev_cp() -> List[int]:
    return bit_rev_cp(list(range(8)))


# @pytest.mark.skip
def test_bit_rev_cp(three_bit_rev_cp, expected_three_bit_rev):
    assert three_bit_rev_cp == expected_three_bit_rev


@pytest.fixture
def make_some_centralized_integers() -> List[Tuple[int, int]]:
    return [(i, cent(q=small_q_for_testing, halfmod=small_halfmod_for_testing, logmod=small_logmod_for_testing, val=i))
            for i in range(small_q_for_testing)]


@pytest.fixture
def expected_centralized_integers() -> List[Tuple[int, int]]:
    return [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, -8), (10, -7), (11, -6),
            (12, -5), (13, -4), (14, -3), (15, -2), (16, -1)]


# @pytest.mark.skip
def test_cent(make_some_centralized_integers, expected_centralized_integers):
    assert make_some_centralized_integers == expected_centralized_integers


@pytest.fixture
def some_small_zetas_and_invs() -> Tuple[List[int], List[int]]:
    return make_zetas_and_invs(
        q=small_q_for_testing,
        d=small_d_for_testing,
        halfmod=small_halfmod_for_testing,
        logmod=small_logmod_for_testing,
        n=small_n_for_testing,
        lgn=small_lgn_for_testing
    )


@pytest.fixture
def expected_small_zetas_and_invs() -> Tuple[List[int], List[int]]:
    return [-1, -4, -8, 3], [-1, 4, 2, 6]


# @pytest.mark.skip
def test_make_zetas_and_invs(some_small_zetas_and_invs, expected_small_zetas_and_invs):
    assert some_small_zetas_and_invs == expected_small_zetas_and_invs


@pytest.fixture
def zetas_and_invs_for_testing() -> Tuple[List[int], List[int]]:
    return make_zetas_and_invs(
        q=modulus_for_testing,
        d=degree_for_testing,
        halfmod=halfmod_for_testing,
        logmod=logmod_for_testing,
        n=n_for_testing,
        lgn=lgn_for_testing
    )


@pytest.fixture
def seven() -> List[int]:
    return [7] + [0 for _ in range(n_for_testing - 1)]


@pytest.fixture
def seven_plus_x() -> List[int]:
    return [7, 1] + [0 for _ in range(n_for_testing - 2)]


@pytest.fixture
def ntt_seven(seven, zetas_and_invs_for_testing) -> List[int]:
    zetas, zeta_invs = zetas_and_invs_for_testing
    return ntt(q=modulus_for_testing, zetas=zetas, zetas_inv=zeta_invs, inv_flag=False, halfmod=halfmod_for_testing,
               logmod=logmod_for_testing, n=n_for_testing, lgn=lgn_for_testing, val=seven)


@pytest.fixture
def ntt_seven_plus_x(seven_plus_x, zetas_and_invs_for_testing) -> List[int]:
    zetas, zeta_invs = zetas_and_invs_for_testing
    return ntt(q=modulus_for_testing, zetas=zetas, zetas_inv=zeta_invs, inv_flag=False, halfmod=halfmod_for_testing,
               logmod=logmod_for_testing, n=n_for_testing, lgn=lgn_for_testing, val=seven_plus_x)


@pytest.fixture
def expected_ntt_seven(seven) -> List[int]:
    return [7 for _ in range(n_for_testing)]


@pytest.fixture
def expected_ntt_seven_plus_x(seven_plus_x, prim_rou_and_inv_for_testing) -> List[int]:
    w, w_inv = prim_rou_and_inv_for_testing
    return [cent(q=modulus_for_testing, halfmod=halfmod_for_testing, logmod=logmod_for_testing, val=7 + w ** i) for i in
            range(n_for_testing)]


@pytest.fixture
def intt_of_ntt_seven(ntt_seven, zetas_and_invs_for_testing) -> List[int]:
    zetas, zeta_invs = zetas_and_invs_for_testing
    return ntt(q=modulus_for_testing, zetas=zetas, zetas_inv=zeta_invs, inv_flag=True, halfmod=halfmod_for_testing,
               logmod=logmod_for_testing, n=n_for_testing, lgn=lgn_for_testing, val=ntt_seven)


@pytest.fixture
def intt_of_ntt_seven_plus_x(ntt_seven_plus_x, zetas_and_invs_for_testing) -> List[int]:
    zetas, zeta_invs = zetas_and_invs_for_testing
    return ntt(q=modulus_for_testing, zetas=zetas, zetas_inv=zeta_invs, inv_flag=True, halfmod=halfmod_for_testing,
               logmod=logmod_for_testing, n=n_for_testing, lgn=lgn_for_testing, val=ntt_seven_plus_x)


# @pytest.mark.skip
def test_ntt(prim_rou_and_inv_for_testing, zetas_and_invs_for_testing, seven, seven_plus_x, ntt_seven,
             expected_ntt_seven, intt_of_ntt_seven, intt_of_ntt_seven_plus_x):
    assert ntt_seven == expected_ntt_seven
    assert intt_of_ntt_seven == seven
    assert intt_of_ntt_seven_plus_x == seven_plus_x


# @pytest.mark.skip
def test_binary_digest():
    """
    TODO: Mock SHAKE256 for testing this. Write later.
    """
    pass


# @pytest.mark.skip
def test_decode2coef():
    next_secpar: int = 1
    next_bd: int = 1
    assert decode2coef(secpar=next_secpar, bd=next_bd, val='0') == -1
    assert decode2coef(secpar=next_secpar, bd=next_bd, val='1') == 1
    next_bd = 2
    assert decode2coef(secpar=next_secpar, bd=next_bd, val='000') == -1
    assert decode2coef(secpar=next_secpar, bd=next_bd, val='001') == -2
    assert decode2coef(secpar=next_secpar, bd=next_bd, val='010') == -1
    assert decode2coef(secpar=next_secpar, bd=next_bd, val='011') == -2
    assert decode2coef(secpar=next_secpar, bd=next_bd, val='100') == 1
    assert decode2coef(secpar=next_secpar, bd=next_bd, val='101') == 2
    assert decode2coef(secpar=next_secpar, bd=next_bd, val='110') == 1
    assert decode2coef(secpar=next_secpar, bd=next_bd, val='111') == 2
    next_secpar = 128
    next_bd = 3
    num_bits = ceil(log2(next_bd)) + 1 + next_secpar
    negative_one: str = bin(0)[2:].zfill(num_bits)
    assert -1 == decode2coef(secpar=next_secpar, bd=next_bd, val=negative_one)
    negative_two: str = negative_one[:-1] + '1'
    assert -2 == decode2coef(secpar=next_secpar, bd=next_bd, val=negative_two)
    negative_three: str = negative_two[:-2] + '10'
    assert -3 == decode2coef(secpar=next_secpar, bd=next_bd, val=negative_three)
    another_negative_one: str = negative_three[:-1] + '1'
    assert -1 == decode2coef(secpar=next_secpar, bd=next_bd, val=another_negative_one)
    another_negative_two: str = another_negative_one[:-3] + '100'
    assert -2 == decode2coef(secpar=next_secpar, bd=next_bd, val=another_negative_two)
    another_negative_three: str = another_negative_two[:-1] + '1'
    assert -3 == decode2coef(secpar=next_secpar, bd=next_bd, val=another_negative_three)
    a_one: str = '1' + negative_one[1:]
    assert 1 == decode2coef(secpar=next_secpar, bd=next_bd, val=a_one)
    a_two: str = a_one[:-1] + '1'
    assert 2 == decode2coef(secpar=next_secpar, bd=next_bd, val=a_two)
    a_three: str = a_two[:-2] + '10'
    assert 3 == decode2coef(secpar=next_secpar, bd=next_bd, val=a_three)
    another_one: str = a_three[:-1] + '1'
    assert 1 == decode2coef(secpar=next_secpar, bd=next_bd, val=another_one)
    another_two: str = another_one[:-3] + '100'
    assert 2 == decode2coef(secpar=next_secpar, bd=next_bd, val=another_two)
    another_three: str = another_two[:-1] + '1'
    assert 3 == decode2coef(secpar=next_secpar, bd=next_bd, val=another_three)


# @pytest.mark.skip
def test_decode2coefs():
    assert decode2coefs(secpar=1, bd=2, wt=2, val='110111') == [1, 2]


# @pytest.mark.skip
def test_decode2indices():
    assert decode2indices(secpar=1, d=4, wt=2, val='00000') == [0, 1]
    assert decode2indices(secpar=1, d=4, wt=2, val='00001') == [0, 2]
    assert decode2indices(secpar=1, d=4, wt=2, val='00010') == [0, 3]
    assert decode2indices(secpar=1, d=4, wt=2, val='00011') == [0, 1]
    assert decode2indices(secpar=1, d=4, wt=2, val='00100') == [0, 2]
    assert decode2indices(secpar=1, d=4, wt=2, val='00101') == [0, 3]
    assert decode2indices(secpar=1, d=4, wt=2, val='00110') == [0, 1]
    assert decode2indices(secpar=1, d=4, wt=2, val='00111') == [0, 2]
    assert decode2indices(secpar=1, d=4, wt=2, val='01000') == [1, 0]
    assert decode2indices(secpar=1, d=4, wt=2, val='01001') == [1, 2]
    assert decode2indices(secpar=1, d=4, wt=2, val='01010') == [1, 3]
    assert decode2indices(secpar=1, d=4, wt=2, val='01011') == [1, 0]
    assert decode2indices(secpar=1, d=4, wt=2, val='01100') == [1, 2]
    assert decode2indices(secpar=1, d=4, wt=2, val='01101') == [1, 3]
    assert decode2indices(secpar=1, d=4, wt=2, val='01110') == [1, 0]
    assert decode2indices(secpar=1, d=4, wt=2, val='01111') == [1, 2]
    assert decode2indices(secpar=1, d=4, wt=2, val='10000') == [2, 0]
    assert decode2indices(secpar=1, d=4, wt=2, val='10001') == [2, 1]
    assert decode2indices(secpar=1, d=4, wt=2, val='10010') == [2, 3]
    assert decode2indices(secpar=1, d=4, wt=2, val='10011') == [2, 0]
    assert decode2indices(secpar=1, d=4, wt=2, val='10100') == [2, 1]
    assert decode2indices(secpar=1, d=4, wt=2, val='10101') == [2, 3]
    assert decode2indices(secpar=1, d=4, wt=2, val='10110') == [2, 0]
    assert decode2indices(secpar=1, d=4, wt=2, val='10111') == [2, 1]
    assert decode2indices(secpar=1, d=4, wt=2, val='11000') == [3, 0]
    assert decode2indices(secpar=1, d=4, wt=2, val='11001') == [3, 1]
    assert decode2indices(secpar=1, d=4, wt=2, val='11010') == [3, 2]
    assert decode2indices(secpar=1, d=4, wt=2, val='11011') == [3, 0]
    assert decode2indices(secpar=1, d=4, wt=2, val='11100') == [3, 1]
    assert decode2indices(secpar=1, d=4, wt=2, val='11101') == [3, 2]
    assert decode2indices(secpar=1, d=4, wt=2, val='11110') == [3, 0]
    assert decode2indices(secpar=1, d=4, wt=2, val='11111') == [3, 1]


# @pytest.mark.skip
def test_decode2polycoefs():
    assert decode2polycoefs(secpar=1, d=4, bd=2, wt=2, val='11110110111') == {3: 1, 0: 2}


@pytest.fixture
def lp_for_testing() -> LatticeParameters:
    return LatticeParameters(pars=pars_for_testing)


# @pytest.mark.skip
def test_lattice_parameters(prim_rou_and_inv_for_testing, zetas_and_invs_for_testing, lp_for_testing):
    w, w_inv = prim_rou_and_inv_for_testing
    zetas, zeta_invs = zetas_and_invs_for_testing
    foo = 'foo'
    bar = 0
    baz = -1
    assert LatticeParameters(pars=pars_for_testing) == lp_for_testing
    assert lp_for_testing.degree == degree_for_testing
    assert lp_for_testing.length == length_for_testing
    assert lp_for_testing.modulus == modulus_for_testing
    assert lp_for_testing.rou == w
    assert lp_for_testing.rou_inv == w_inv
    assert lp_for_testing.zetas == zetas
    assert lp_for_testing.zetas_invs == zeta_invs

    bad_pars_for_testing: dict = deepcopy(pars_for_testing)
    # Test failure of non-integer degree
    bad_pars_for_testing['degree'] = foo
    with pytest.raises(ValueError):
        LatticeParameters(pars=bad_pars_for_testing)

    # Test failure of zero degree
    bad_pars_for_testing['degree'] = bar
    with pytest.raises(ValueError):
        LatticeParameters(pars=bad_pars_for_testing)

    # Test failure of negative degree
    bad_pars_for_testing['degree'] = baz
    with pytest.raises(ValueError):
        LatticeParameters(pars=bad_pars_for_testing)

    # reset
    bad_pars_for_testing['degree'] = degree_for_testing

    # Test failure of non-integer length
    bad_pars_for_testing['length'] = foo
    with pytest.raises(ValueError):
        LatticeParameters(pars=bad_pars_for_testing)

    # Test failure of zero length
    bad_pars_for_testing['length'] = bar
    with pytest.raises(ValueError):
        LatticeParameters(pars=bad_pars_for_testing)

    # Test failure of negative length
    bad_pars_for_testing['length'] = baz
    with pytest.raises(ValueError):
        LatticeParameters(pars=bad_pars_for_testing)

    # reset
    bad_pars_for_testing['length'] = length_for_testing

    # Test failure of non-integer modulus
    bad_pars_for_testing['modulus'] = foo
    with pytest.raises(ValueError):
        LatticeParameters(pars=bad_pars_for_testing)

    # Test failure of zero modulus
    bad_pars_for_testing['modulus'] = bar
    with pytest.raises(ValueError):
        LatticeParameters(pars=bad_pars_for_testing)

    # Test failure of negative modulus
    bad_pars_for_testing['modulus'] = baz
    with pytest.raises(ValueError):
        LatticeParameters(pars=bad_pars_for_testing)

    # Test failure of unit modulus
    bad_pars_for_testing['modulus'] = -baz
    with pytest.raises(ValueError):
        LatticeParameters(pars=bad_pars_for_testing)


# @pytest.mark.skip
def test_lattice_repr(lp_for_testing):
    assert str(lp_for_testing) == str((degree_for_testing, length_for_testing, modulus_for_testing))


# @pytest.mark.skip
def test_lattice_eq(lp_for_testing):
    assert LatticeParameters(pars=pars_for_testing) == lp_for_testing


# @pytest.mark.skip
def test_get_gen_bits_per_poly(lp_for_testing):
    assert get_gen_bits_per_poly(secpar=1, lp=lp_for_testing, wt=1, bd=1) == ceil(
        (ceil(log2(lp_for_testing.degree)) + 1 + ceil(log2(1)) + 1) / 8.0)
    assert get_gen_bits_per_poly(secpar=1, lp=lp_for_testing, wt=2, bd=2) == ceil(
        (ceil(log2(lp_for_testing.degree)) + (log2(lp_for_testing.degree) + 1) + 1 + ceil(log2(1)) + 1) / 8.0)


@pytest.fixture
def one(lp_for_testing) -> Polynomial:
    return Polynomial(pars=lp_for_testing, coefs={0: 1})


@pytest.fixture
def some_ran_lin_polys(lp_for_testing) -> List[Tuple[int, int, Polynomial]]:
    result = []
    for i in range(2 * sample_size_for_random_tests):
        a = (2 * randbits(1) - 1) * (randbelow(lp_for_testing.halfmod) + 1)
        b = (2 * randbits(1) - 1) * (randbelow(lp_for_testing.halfmod) + 1)
        result += [(a, b, Polynomial(pars=lp_for_testing, coefs={0: a, 1: b}))]
    return result


# @pytest.mark.skip
def test_polynomial_init(prim_rou_and_inv_for_testing, lp_for_testing, one, some_ran_lin_polys):
    lp: LatticeParameters = lp_for_testing
    w, w_inv = lp.rou, lp.rou_inv
    assert w, w_inv == prim_rou_and_inv_for_testing

    # First, let's mess with the identity polynomial
    assert one.ntt_representation == [1 for _ in range(n_for_testing)]

    # Now let's make some random linear polynomials
    for next_tuple in some_ran_lin_polys:
        a, b, f = next_tuple
        assert -lp.halfmod <= a <= lp.halfmod
        assert -lp.halfmod <= b <= lp.halfmod
        assert isinstance(f, Polynomial)
        assert f.ntt_representation == [
            cent(q=lp.modulus, halfmod=lp.halfmod, logmod=lp.logmod, val=a + b * w ** k) for k in range(lp.n)
        ]


# @pytest.mark.skip
def test_polynomial_eq(lp_for_testing, one, some_ran_lin_polys):
    lp: LatticeParameters = lp_for_testing

    # First, let's make two identity polynomials and check they are equal.
    another_one = Polynomial(pars=lp, coefs={0: 1})
    assert one == another_one

    # Now check that if we change a single coefficient, the result changes
    not_one = Polynomial(pars=lp, coefs={0: -1})
    assert one != not_one

    # Now let's do the same with some random linear polynomials
    for next_tuple in some_ran_lin_polys:
        a, b, next_poly = next_tuple
        another_poly = Polynomial(pars=lp, coefs={0: a, 1: b})
        assert next_poly == another_poly

        # Now check that if we change a single coefficient, the result changes
        not_another_poly = Polynomial(pars=lp, coefs={0: a, 1: -b})
        assert next_poly != not_another_poly


@pytest.fixture
def two(lp_for_testing) -> Polynomial:
    return Polynomial(pars=lp_for_testing, coefs={0: 2})


@pytest.fixture
def pairs_ran_lin_poly(some_ran_lin_polys) -> List[Tuple[Tuple[int, int, Polynomial], Tuple[int, int, Polynomial]]]:
    result = []
    for i in range(0, sample_size_for_random_tests, 2):
        result += [(some_ran_lin_polys[i], some_ran_lin_polys[i + 1])]
    return result


@pytest.fixture
def pairs_of_random_polys_and_their_sums(
        lp_for_testing,
        pairs_ran_lin_poly
) -> List[
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
        obs_h_coefs, obs_h_norm, obs_h_wt = observed_h.coefficient_representation_and_norm_and_weight()
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
        expected_h = Polynomial(pars=lp_for_testing, coefs=expected_h_coefs)
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
def test_polynomial_add(lp_for_testing, one, two, pairs_of_random_polys_and_their_sums):
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
def pairs_of_random_polys_and_their_diffs(lp_for_testing, pairs_ran_lin_poly) -> \
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
        obs_h_coefs, obs_h_norm, obs_h_wt = observed_h.coefficient_representation_and_norm_and_weight()
        expected_h = Polynomial(pars=lp_for_testing, coefs=expected_h_coefs)
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
def test_polynomial_sub(lp_for_testing, pairs_of_random_polys_and_their_diffs):
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
def pairs_of_random_polys_and_their_products(lp_for_testing, pairs_ran_lin_poly) -> \
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
        obs_h_coefs, obs_h_norm, obs_h_wt = observed_h.coefficient_representation_and_norm_and_weight()
        expected_h = Polynomial(pars=lp_for_testing, coefs=exp_h_coefs)
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
def test_polynomial_mul_small(lp_for_testing, one, pairs_of_random_polys_and_their_products):
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
        coef_rep, norm, wt = f.coefficient_representation_and_norm_and_weight()
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
def test_polynomial_reset_vals(lp_for_testing, one):
    x = deepcopy(one)
    x._reset_vals(coefs={0: 2})
    assert x.ntt_representation == [2 for _ in range(lp_for_testing.n)]
    x._reset_vals(coefs={0: 1})
    assert x == one


# @pytest.mark.skip
def test_polynomial_get_coefs(lp_for_testing, one):
    x = Polynomial(pars=lp_for_testing, coefs={1: 1})
    f = one + x
    assert f.coefficient_representation_and_norm_and_weight() == ({0: 1, 1: 1}, 1, 2)


# @pytest.mark.skip
def test_polynomial_norm_and_weight(some_ran_lin_polys):
    for next_item in some_ran_lin_polys:
        a, b, f = next_item
        f_coefs, n, w = f.coefficient_representation_and_norm_and_weight()
        assert n == max(abs(a), abs(b))
        if a != 0 and b != 0:
            assert w == 2
        elif (a == 0 and b != 0) or (a != 0 and b == 0):
            assert w == 1
        else:
            assert w == 0


# @pytest.mark.skip
def test_rand_poly(lp_for_testing):
    f = randpoly(lp=lp_for_testing, bd=1, wt=2)
    assert isinstance(f, Polynomial)
    f_coefs, n, w = f.coefficient_representation_and_norm_and_weight()
    assert n <= 1 and w <= 2
    assert max(abs(f_coefs[i]) for i in f_coefs) <= n
    assert len(f_coefs) <= w


@pytest.fixture
def some_random_polys_for_a_vector(lp_for_testing) -> List[Polynomial]:
    lp: LatticeParameters = lp_for_testing
    return [randpoly(lp=lp, bd=lp.halfmod // 2, wt=lp.degree // 2) for _ in range(lp.length)]


# @pytest.mark.skip
def test_polynomial_vector_init(lp_for_testing, some_random_polys_for_a_vector):
    lp: LatticeParameters = lp_for_testing
    assert PolynomialVector(pars=lp, entries=some_random_polys_for_a_vector)
    v = PolynomialVector(pars=lp, entries=some_random_polys_for_a_vector)
    tmp = v.coefficient_representation_and_norm_and_weight()
    assert max(i[1] for i in tmp) <= lp.halfmod // 2
    assert max(i[2] for i in tmp) <= lp.degree // 2
    assert randpolyvec(lp=lp, bd=norm_for_testing, wt=weight_for_testing)


@pytest.fixture
def some_random_polyvec(lp_for_testing, some_random_polys_for_a_vector) -> PolynomialVector:
    return PolynomialVector(pars=lp_for_testing, entries=some_random_polys_for_a_vector)


@pytest.fixture
def some_random_polyvecs(lp_for_testing, some_random_polys_for_a_vector) -> List[PolynomialVector]:
    return [PolynomialVector(pars=lp_for_testing, entries=some_random_polys_for_a_vector) for _ in
            range(sample_size_for_random_tests)]


# @pytest.mark.skip
def test_polynomial_vector_eq(lp_for_testing, some_random_polys_for_a_vector, some_random_polyvec):
    lp: LatticeParameters = lp_for_testing
    v = PolynomialVector(pars=lp, entries=deepcopy(some_random_polys_for_a_vector))
    assert v == some_random_polyvec


@pytest.fixture
def some_ran_polyvec_pairs_sums(lp_for_testing) -> List[Tuple[PolynomialVector, PolynomialVector, PolynomialVector]]:
    result = []
    while len(result) < sample_size_for_random_tests:
        f = randpolyvec(lp=lp_for_testing, bd=lp_for_testing.halfmod, wt=lp_for_testing.degree)
        g = randpolyvec(lp=lp_for_testing, bd=lp_for_testing.halfmod, wt=lp_for_testing.degree)
        h = f + g
        result += [(f, g, h)]
    return result


@pytest.fixture
def some_ran_polyvec_pairs_diffs(lp_for_testing) -> List[Tuple[PolynomialVector, PolynomialVector, PolynomialVector]]:
    result = []
    while len(result) < sample_size_for_random_tests:
        f = randpolyvec(lp=lp_for_testing, bd=lp_for_testing.halfmod, wt=lp_for_testing.degree)
        g = randpolyvec(lp=lp_for_testing, bd=lp_for_testing.halfmod, wt=lp_for_testing.degree)
        h = f - g
        result += [(f, g, h)]
    return result


@pytest.fixture
def some_ran_scaling(lp_for_testing) -> List[Tuple[PolynomialVector, PolynomialVector, PolynomialVector]]:
    result = []
    while len(result) < sample_size_for_random_tests:
        f = randpoly(lp=lp_for_testing, bd=lp_for_testing.halfmod, wt=lp_for_testing.degree)
        g = randpolyvec(lp=lp_for_testing, bd=lp_for_testing.halfmod, wt=lp_for_testing.degree)
        h = g ** f
        result += [(f, g, h)]
    return result


# @pytest.mark.skip
def test_polynomial_vector_add(lp_for_testing, some_ran_polyvec_pairs_sums):
    # TODO: Modify to exploit fixtures more effectively, this is spaghetti
    lp: LatticeParameters = lp_for_testing
    for next_item in some_ran_polyvec_pairs_sums:
        f, g, observed_h = next_item
        for i, val in enumerate(zip(f.entries, g.entries, observed_h.entries)):
            ff, gg, hh = val
            diff = [ff.ntt_representation[j] + gg.ntt_representation[j] for j in range(lp.degree)]
            diff = [diff[j] - hh.ntt_representation[j] for j in range(lp.degree)]
            assert all(
                cent(q=lp.modulus, halfmod=lp.halfmod, logmod=lp.logmod, val=diff[j]) == 0 for j in range(lp.degree)
            )
        observed_h_coefs_and_norms_and_wts = observed_h.coefficient_representation_and_norm_and_weight()
        f_coefs_and_norms_and_wts = f.coefficient_representation_and_norm_and_weight()
        g_coefs_and_norms_and_wts = g.coefficient_representation_and_norm_and_weight()
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
def test_polynomial_vector_sub(lp_for_testing, some_ran_polyvec_pairs_diffs):
    lp: LatticeParameters = lp_for_testing
    for next_item in some_ran_polyvec_pairs_diffs:
        f, g, observed_h = next_item
        observed_h_coefs_and_norms_and_weights = [
            i.coefficient_representation_and_norm_and_weight() for i in observed_h.entries
        ]
        for i in range(lp.length):
            f_coefs, f_norm, f_wt = f.entries[i].coefficient_representation_and_norm_and_weight()
            g_coefs, g_norm, g_wt = g.entries[i].coefficient_representation_and_norm_and_weight()
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
def test_polynomial_vector_mul(lp_for_testing, one, some_random_polyvecs):
    # tests the dot product
    lp: LatticeParameters = lp_for_testing
    all_ones: PolynomialVector = PolynomialVector(pars=lp, entries=[deepcopy(one) for _ in range(lp.length)])
    for v in some_random_polyvecs:
        expected_sum: Polynomial = sum(x for x in v.entries)
        observed_sum: Polynomial = all_ones * v
        assert observed_sum == expected_sum
        observed_sum: Polynomial = v * all_ones
        assert observed_sum == expected_sum


@pytest.fixture
def some_random_linear_polyvecs(lp_for_testing) -> list:
    result = []
    while len(result) < sample_size_for_random_tests:
        next_result_entry = []
        entries = []
        while len(entries) < lp_for_testing.length:
            a: int = (2 * randbits(1) - 1) * (randbelow(lp_for_testing.halfmod) + 1)
            b: int = (2 * randbits(1) - 1) * (randbelow(lp_for_testing.halfmod) + 1)
            if a != 0 or b != 0:
                next_poly: Polynomial = Polynomial(pars=lp_for_testing, coefs={0: a, 1: b})
                next_result_entry += [(a, b, next_poly)]
                entries += [next_poly]
        next_polyvec = PolynomialVector(pars=lp_for_testing, entries=entries)
        next_result_entry += [next_polyvec]
        result += [next_result_entry]
    return result


@pytest.fixture
def expected_polyvec_rep(some_random_linear_polyvecs) -> List[str]:
    result = []
    for next_polyvec_data in some_random_linear_polyvecs:
        next_rep: str = '['
        the_polyvec = next_polyvec_data[-1]  # last entry of next_polyvec_data is the polynomial vector itself
        for i, val in enumerate(zip(the_polyvec.entries, next_polyvec_data[:-1])):
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
            # assert the_next_poly.coefficient_representation_and_norm_and_weight() == the_coefs, the_norm, the_wt
            # assert also_the_next_poly == the_next_poly
            the_coefs_sorted_keys = sorted(list(the_coefs.keys()))
            sorted_coefs = [(i, the_coefs[i]) for i in the_coefs_sorted_keys]
            next_rep += str((sorted_coefs, the_norm, the_wt)) + ', '
        next_rep = next_rep[:-2] + ']'
        result += [next_rep]
    return result


# @pytest.mark.skip
def test_polynomial_vector_repr(some_random_linear_polyvecs, expected_polyvec_rep):
    for next_pair in zip(some_random_linear_polyvecs, expected_polyvec_rep):
        next_random_linear_polyvec, next_expected_rep = next_pair
        assert str(next_random_linear_polyvec[-1]) == next_expected_rep


# @pytest.mark.skip
def test_polynomial_vector_pow(lp_for_testing):
    lp: LatticeParameters = lp_for_testing

    for k in range(sample_size_for_random_tests):
        v: Polynomial = randpoly(lp=lp, bd=17, wt=23)
        u: PolynomialVector = randpolyvec(lp=lp, bd=7176, wt=384)
        expected_scaled_vector: PolynomialVector = deepcopy(u)
        expected_scaled_vector.entries = [i * v for i in expected_scaled_vector.entries]
        observed_scaled_vector = u ** v
        assert expected_scaled_vector == observed_scaled_vector


# @pytest.mark.skip
def test_polyvec_coefficient_representation_and_norm_and_weight(lp_for_testing):
    f: PolynomialVector = randpolyvec(lp=lp_for_testing, bd=7176, wt=384)
    assert isinstance(f, PolynomialVector)
    result = f.coefficient_representation_and_norm_and_weight()
    for i in result:
        coef_rep, n, w = i
        assert n <= 7176
        assert w <= 384
        assert w == len(coef_rep)
        assert n == max(abs(coef_rep[i]) for i in coef_rep)


# @pytest.mark.skip
def test_randpolyvec(lp_for_testing):
    f: PolynomialVector = randpolyvec(lp=lp_for_testing, bd=7176, wt=384)
    assert isinstance(f, PolynomialVector)
    results = f.coefficient_representation_and_norm_and_weight()
    for i in results:
        coef_rep, n, w = i
        assert n <= 7176
        assert w <= 384
        assert len(coef_rep) == w
        assert n == max(abs(coef_rep[i]) for i in coef_rep)


# @pytest.mark.skip
def test_decode_bitstring_to_coefficient():
    # TODO: Unnecessarily thorough without keeping track of touched bounds, but doesn't test what is intended to be
    #  tested by avoiding touched bounds this way... more clever test needed
    secpar = 8
    touched_bounds = dict()
    for bound in range(1, small_q_for_testing // 2):
        if bound not in touched_bounds:
            touched_bounds[bound] = 1
            if bound == 1:
                outbits = 1
                for expected_result in [-1, 1]:
                    bitstring = bin((1 + expected_result) // 2)[2:]
                    assert len(bitstring) == outbits
                    observed_result = decode2coef(secpar, bound, bitstring)
                    assert expected_result == observed_result
            else:
                outbits = ceil(log2(bound)) + 1 + secpar
                for test_int in range(bound):
                    for test_sign_bit in [0, 1]:
                        expected_result = (2 * test_sign_bit - 1) * (1 + test_int)
                        bitstring = str(test_sign_bit) + bin(test_int)[2:].zfill(outbits - 1)
                        assert len(bitstring) == outbits
                        observed_result = decode2coef(secpar, bound, bitstring)
                        assert expected_result == observed_result


# @pytest.mark.skip
def test_decode_bitstring_to_indices():
    secpar = 8
    degree = 8
    weight = 3
    # say we want [1, 0, 0, 1, 0, 0, 1, 0]
    # first index is 0, so the first part of the bitstring is '000' (only need 3 bits for the first)
    # second index is 3, requires ceil(log2(8)) + secpar = 11 bits since we already picked an index.
    # but the remaining indices are [1, 2, 3, 4, 5, 6, 7]
    # to get 3 out of this, we need to access index 2!!!
    # so we need an 11-bit integer == 2 mod 7. for fun, let's use 2 + 2**6 * 7 = 450 -> '00111000010`
    # third index is 6, requires ceil(log2(8)) + secpar = 11 bits since we already picked two indices
    # but the remaining indices are [1, 2, 4, 5, 6, 7]
    # to get 6 out of this, we need to access index 4
    # let's use 4 + 2**5 * 6 = 196 -> '00011000100'
    # i.e. set our bitstring = '0000011100001000011000100' and we should get
    # [0, 3, 6].
    test_bitstring = '0000011100001000011000100'
    expected_result = [0, 3, 6]
    observed_result = decode2indices(secpar, degree, weight, test_bitstring)
    assert expected_result == observed_result


# @pytest.mark.skip
def test_decode_bitstring_to_polynomial_coefficients():
    secpar = 8
    degree = 8
    weight = 3
    # Let's test the special case that the bound is 1
    bound = 1
    # Let's construct the bitstring that should give us {0: 1, 3: 1, 6: -1}.
    expected_result = {0: 1, 3: 1, 6: -1}
    # expected coefficients will be [1, 1, -1]
    # expected indices will be [0, 3, 6]
    # See test for indices, use '0000011100001000011000100'
    test_bitstring_for_indices = '0000011100001000011000100'
    # to get the coef 1, we decode from the bit 1, and to get the coef -1, we decode from the bit -1
    test_bitstring_for_coefs = '110'
    test_bitstring = test_bitstring_for_indices + test_bitstring_for_coefs
    observed_result = decode2polycoefs(secpar, degree, bound, weight, test_bitstring)
    assert expected_result == observed_result

    # Let's test the same expected_result but when the bound is 2
    bound = 2
    # since 1 + ceil(log2(bound)) + secpar = 10, we need 10 bits for each coef, first bit being the presign
    # for the coef 1, we need an integer x such that 1 + (x % bound) == 1, so x % bound == 0
    # so any even integer with at most 9 bits will work when bound = 2
    # let's say x = 2**6, expanded to 9 bits, is '001000000' in binary
    # then we prefix with the presign -> '1001000000'
    # for the coef -1, the magnitude is some integer x such that 1 + (x % bound) == 1 again
    # so again any even integer with at most 9 bits will work when bound = 2
    # let's say x = 2**8 + 2**7 for fun -> '110000000'
    # then since the presign of -1 is -1, which is decoded from the bit 0, we prefix with 0 -> '0110000000'
    test_bitstring_for_coefs = '100100000010010000000110000000'
    test_bitstring = test_bitstring_for_indices + test_bitstring_for_coefs
    observed_result = decode2polycoefs(secpar, degree, bound, weight, test_bitstring)
    assert expected_result == observed_result


# @pytest.mark.skip
def test_decode_bitstring_to_coefficients():
    # Thorough test
    secpar = 8
    touched_bound_weight_pairs = dict()
    for bound in range(1, modulus_for_testing // 2):
        for weight in range(1, degree_for_testing):
            if (bound, weight) in touched_bound_weight_pairs:
                touched_bound_weight_pairs[(bound, weight)] = 1
                if bound > 1:
                    expected_result = list()
                    while len(expected_result) < weight:
                        next_int_sign = 2 * randbits(1) - 1
                        next_int_mag = 1 + randbelow(bound)
                        expected_result += [next_int_sign * next_int_mag]
                    outbits = weight * (1 + ceil(log2(bound)) + secpar)
                    bitstrings = list()
                    for i in expected_result:
                        tmp = bin(int(i > 0))[2:] + bin(abs(i) - 1)[2:].zfill(outbits // weight - 1)
                        bitstrings += [tmp]
                    assert len(bitstrings) == weight
                    for i in bitstrings:
                        assert len(i) == outbits // weight
                    merged_bitstrings = ''
                    for i in bitstrings:
                        merged_bitstrings += i
                    observed_result = decode2coefs(secpar, bound, weight, merged_bitstrings)
                    assert expected_result == observed_result


# @pytest.mark.skip
def test_hash2bddpoly():
    # TODO: High priority test, mock shake256 for this
    pass


# @pytest.mark.skip
def test_hash2bddpolyvec():
    # TODO: High priority test, mock shake256 for this
    pass
