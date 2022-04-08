"""
The lattice_algebra module is an (unaudited) prototype containing basic algebraic infrastructure for employing lattice-
based crypto schemes in the Ring (aka Module or Ideal) Short Integer Solution and Ring (Module, Ideal) Learning With
Errors settings where secret/error vectors are a uniformly distributed from the subset of vectors with bounded infinity
norm and constant weight.

Todo
 1. Modify decode2coefs, decode2indices, decode2polycoefs to be more pythonic if possible
 2. Add const_time attribute to Polynomials, add support for non-const-time arithmetic

Documentation
-------------

Documentation is hosted at [https://geometry-labs.github.io/lattice-algebra/](https://geometry-labs.github.io/lattice-algebra/)

Hosting
-------

The repository for this project can be found at [https://www.github.com/geometry-labs/lattice-algebra](https://www.github.com/geometry-labs/lattice-algebra).

License
-------

Released under the MIT License, see LICENSE file for details.

Copyright
---------

Copyright (c) 2022 Geometry Labs, Inc.
Funded by The QRL Foundation.

Contributors
------------
Brandon Goodell (lead author), Mitchell Krawiec-Thayer, Rob Cannon.

"""
from math import ceil, sqrt, log2
from copy import deepcopy
from secrets import randbits
from hashlib import shake_256 as shake
from typing import List, Dict, Tuple, Union


def bits_to_indices(secpar: int, degree: int, wt: int) -> int:
    return ceil(log2(degree)) + (wt - 1) * (ceil(log2(degree)) + secpar)


def bits_to_decode(secpar: int, bd: int) -> int:
    return ceil(log2(bd)) + 1 + secpar


def is_prime(val: int) -> bool:
    """
    Test whether input integer is prime with the Sieve of Eratosthenes.

    :param val: Input value
    :type val: int

    :return: Indicate whether q is prime.
    :rtype: bool
    """
    return all([val % i != 0 for i in range(2, ceil(sqrt(val)) + 1)])


def is_pow_two(val: int) -> bool:
    """
    Test if input integer is power of two by summing the bits in its binary expansion and testing for 1.

    :param val: Input value
    :type val: int

    :return: Indicate whether x is a power-of-two.
    :rtype: bool
    """
    return val > 0 and not (val & (val - 1))


def has_prim_rou(modulus: int, degree: int) -> bool:
    """
    Test whether Z/qZ has a primitive 2d-th root of unity.
    """
    return modulus % (2 * degree) == 1


def is_ntt_friendly_prime(modulus: int, degree: int) -> bool:
    """
    Test whether input integer pair can be used to construct an NTT-friendly ring.

    :param modulus: Input modulus
    :type modulus: int
    :param degree: Input degree
    :type degree: int

    :return: Indicate whether q is prime and q-1 == 0 mod 2d.
    :rtype: bool
    """
    return is_prime(modulus) and is_pow_two(degree) and has_prim_rou(modulus=modulus, degree=degree)


def is_prim_rou(modulus: int, degree: int, val: int) -> bool:
    """
    Test whether input x is a primitive 2d^th root of unity in the integers modulo q. Does not require q and d to be
    NTT-friendly pair.

    :param modulus: Input modulus
    :type modulus: int
    :param degree: Input degree
    :type degree: int
    :param val: Purported root of unity
    :type val: int

    :return: Boolean indicating x**(2d) == 1 and x**i != 1 for 1 <= i < 2d.
    :rtype: bool
    """
    return all(val ** k % modulus != 1 for k in range(1, 2 * degree)) and val ** (2 * degree) % modulus == 1


def get_prim_rou_and_rou_inv(modulus: int, degree: int) -> Union[None, Tuple[int, int]]:
    """
    Compute a primitive 2d-th root of unity modulo q and its inverse. Raises a ValueError if (d, q) are not NTT-
    friendly pair. Works by finding the first (in natural number order) primitive root of unity and its inverse.

    :param modulus: Input modulus
    :type modulus: int
    :param degree: Input degree
    :type degree: int

    :return: Root of unity and its inverse in a tuple.
    :rtype: Tuple[int, int]
    """
    if not (is_ntt_friendly_prime(modulus, degree)):
        raise ValueError('Input q and d are not ntt-friendly prime and degree.')
    # If we do not raise a ValueError, then there exists a primitive root of unity 2 <= x < q.
    x: int = 2
    while x < modulus:
        if is_prim_rou(modulus, degree, x):
            break
        x += 1
    return x, ((x ** (2 * degree - 1)) % modulus)


def is_bitstring(val: str) -> bool:
    """
    Ensure that an ostensible bitstring does not contain any characters besides 0 and 1 (without type conversion)

    :param val: Input to be validated as a bitstring
    :type val: str

    :return: True if the input is a bitstring (i.e. contains only 0's and 1's) and False otherwise.
    :rtype: bool
    """
    if isinstance(val, str):
        return ''.join(sorted(list(set(val)))) in '01'
    return False


touched_bit_rev: Dict[Tuple[int, int], int] = dict()


def bit_rev(num_bits: int, val: int) -> int:
    """
    Reverse the bits in the binary expansion of x and interpret that as an integer.

    :param num_bits: Input length in bits
    :type num_bits: int
    :param val: Input value
    :type val: int

    :return: Output the bit-reversed value of x
    :rtype: int
    """
    if (num_bits, val) not in touched_bit_rev:
        for x in range(2 ** num_bits):
            if (num_bits, x) not in touched_bit_rev:
                x_in_bin: str = bin(x)[2:].zfill(num_bits)
                touched_bit_rev[(num_bits, x)] = int(x_in_bin[::-1], 2)
    return touched_bit_rev[(num_bits, val)]


def bit_rev_cp(val: List[int], n: int) -> List[int]:
    """
    Permute the indices of the input list x to bit-reversed order. Note: does not compute the bit-reverse of the values
    in the input list, just shuffles the input list around.

    :param val: Input values
    :type val: List[int]
    :param n: Length of bit string before copying (the code pre-pends with zeros to get to this length).
    :type n: int

    :return: Output permuted list
    :rtype: List[int]
    """
    if not is_pow_two(len(val)):
        raise ValueError("Can only bit-reverse-copy arrays with power-of-two lengths.")
    return [val[bit_rev(n, i)] for i in range(len(val))]


def cent(q: int, halfmod: int, logmod: int, val: int) -> int:
    """
    Constant-time remainder.

    :param q: Input modulus
    :type q: int
    :param val: Input value
    :type val: int
    :param halfmod: q//2
    :type halfmod: int
    :param logmod: ceil(log2(q))
    :type logmod: int

    :return: Value in the set [-(q//2), ..., q//2] equivalent to x mod q.
    :rtype: int
    """
    y: int = val % q
    intermediate_value: int = y - halfmod - 1
    return y - (1 + (intermediate_value >> logmod)) * q


def make_zetas_and_invs(q: int, d: int, n: int, lgn: int) -> Tuple[List[int], List[int]]:
    """
    Compute powers of primitive root of unity and its inverse for use in the NTT function. Finds the root of unity, say
    zeta, and its inverse, say zeta_inv, then outputs [zeta ** ((2d) // (2**(s+1))) for s in range(log2(2d))] and
    [zeta_inv ** ((2d) // (2**(s+1))) for s in range(log2(2d))], modulo q.

    :param q: Input modulus
    :type q: int
    :param d: Input degree
    :type d: int
    :param n: 2*d
    :type n: int
    :param lgn: ceil(log2(n))
    :type lgn: int

    :return: Return the powers of zeta and zeta_inv for use in NTT, two lists of integers, in a tuple
    :rtype: Tuple[List[int], List[int]]
    """
    powers: List[int] = [n // (2 ** (s + 1)) for s in range(lgn)]
    zeta, zeta_inv = get_prim_rou_and_rou_inv(q, d)
    left: List[int] = [int(zeta ** i) % q for i in powers]
    left = [i if i <= q // 2 else i - q for i in left]
    right: List[int] = [int(zeta_inv ** i) % q for i in powers]
    right = [i if i <= q // 2 else i - q for i in right]
    return left, right


def ntt(q: int, zetas: List[int], zetas_inv: List[int], inv_flag: bool, halfmod: int, logmod: int, n: int, lgn: int,
        val: List[int], const_time_flag: bool = True) -> List[int]:
    """
    Compute the NTT of the input list of integers. Implements the algorithm from Cormen, T. H., Leiserson, C. E.,
    Rivest, R. L., & Stein, C, (2009), "Introduction to algorithms," but replacing exp(-i*2*pi/n) with zeta.

    :param q: Input modulus
    :type q: int
    :param zetas: Input powers of the root of unity
    :type zetas: List[int]
    :param zetas_inv: Input powers of the root of unity
    :type zetas_inv: List[int]
    :param val: Input values
    :type val: List[int]
    :param inv_flag: Indicates whether we are performing forward NTT or inverse NTT
    :type inv_flag: bool
    :param halfmod: q//2
    :type halfmod: int
    :param logmod: ceil(log2(q))
    :type logmod: int
    :param n: 2*d
    :type n: int
    :param lgn: ceil(log2(n))
    :type lgn: int
    :param const_time_flag: Indicates whether arithmetic should be constant-time.
    :type const_time_flag: bool

    :return: Return the NTT (or inverse) of the inputs x.
    :rtype: List[int]
    """
    if sum(int(i) for i in bin(len(val))[2:]) != 1:
        raise ValueError("Can only NTT arrays with lengths that are powers of two.")
    bit_rev_x: List[int] = bit_rev_cp(val=val, n=ceil(log2(len(val))))
    m: int = 1
    for s in range(1, lgn + 1):
        m *= 2
        if inv_flag:
            this_zeta: int = zetas_inv[s - 1]
        else:
            this_zeta: int = zetas[s - 1]
        for k in range(0, n, m):
            w: int = 1
            for j in range(m // 2):
                t: int = w * bit_rev_x[k + j + m // 2]
                u: int = bit_rev_x[k + j]
                if const_time_flag:
                    bit_rev_x[k + j]: int = cent(q=q, halfmod=halfmod, logmod=logmod, val=u + t)
                    bit_rev_x[k + j + m // 2]: int = cent(q=q, halfmod=halfmod, logmod=logmod, val=u - t)
                else:
                    bit_rev_x[k + j]: int = (u + t) % q
                    if bit_rev_x[k + j] > q // 2:
                        bit_rev_x[k + j] = bit_rev_x[k + j] - q
                    bit_rev_x[k + j + m // 2]: int = (u - t) % q
                    if bit_rev_x[k + j + m // 2] > q // 2:
                        bit_rev_x[k + j + m // 2] = bit_rev_x[k + j + m // 2] - q
                w *= this_zeta
    if inv_flag:
        n_inv: int = 1
        while (n_inv * n) % q != 1:
            n_inv += 1
        if const_time_flag:
            bit_rev_x: List[int] = [cent(q=q, halfmod=halfmod, logmod=logmod, val=(n_inv * i)) for i in bit_rev_x]
        else:
            bit_rev_x: List[int] = [(n_inv * i) % q for i in bit_rev_x]
            bit_rev_x = [i if i <= q // 2 else i - q for i in bit_rev_x]
    return bit_rev_x


def binary_digest(msg: str, num_bytes: int, salt: str) -> str:
    """
    Compute input num_bytes bytes from SHAKE256 using the input salt and message.

    :param msg: Input message
    :type msg: str
    :param num_bytes: Input number of bits
    :type num_bytes: int
    :param salt: Input salt
    :type salt: str

    :return: Return the digest of num_bytes bits in binary.
    :rtype: str
    """
    m = shake()
    m.update(salt.encode() + msg.encode())
    return bin(int(m.hexdigest(num_bytes), 16))[2:].zfill(8 * num_bytes)


class LatticeParameters(object):
    """
    Class for handling lattice parameters.

    Attributes
    ----------
        degree: int
            Degree bound for all polynomials.
        length: int
            Length of vectors of polynomials.
        modulus: int
            Modulus for all coefficients
        halfmod: int
            Modulus // 2
        logmod: int
            log2(modulus)
        n: int
            2 * degree
        rou: int
            Primitive 2*degree-th root of unity modulo modulus
        rou_inv: int
            Inverse of rou modulo modulus
        zetas: List[int]
            Powers of rou for use in computing NTT/inverse
        zetas_invs: List[int]
            Powers of rou for use in computing NTT/inverse

    Methods
    -------
        __init__(self)
            Initialize
        __eq__(self, other)
            Check for equality
        __repr__(self)
            String representation of attributes
    """
    degree: int
    length: int
    modulus: int
    halfmod: int
    logmod: int
    n: int
    lgn: int
    rou: int
    rou_inv: int
    zetas: List[int]
    zetas_invs: List[int]

    def __init__(self, degree: int, length: int, modulus: int):
        """
        Create a new LatticeParameters object with input pars, compute the rou, rou_inv, zetas, and zetas_inv

        :param degree: Polynomial degree
        :type degree: int
        :param length: Vector length
        :type length: int
        :param modulus: Prime modulus
        :type length: int
        """
        if degree < 2 or not is_pow_two(val=degree):
            raise ValueError('LatticeParameters requires power-of-two integer degree.')
        elif length < 1:
            raise ValueError('LatticeParameters requires positive integer length.')
        elif modulus < 3 or not is_ntt_friendly_prime(modulus=modulus, degree=degree):
            raise ValueError('LatticeParameters requires NTT-friendly prime modulus-degree pair.')

        self.degree = degree
        self.length = length
        self.modulus = modulus
        self.halfmod = self.modulus // 2
        self.logmod = ceil(log2(self.modulus))
        self.n: int = 2 * self.degree
        self.lgn: int = ceil(log2(self.n))
        self.rou, self.rou_inv = get_prim_rou_and_rou_inv(modulus=self.modulus, degree=self.degree)
        self.zetas, self.zetas_invs = make_zetas_and_invs(q=self.modulus, d=self.degree, n=self.n, lgn=self.lgn)

    def __eq__(self, other) -> bool:
        """
        Compare two LatticeParameters objects for equality. We only check degree, length, and modulus. This is due to
        the fact that all the other parameters are derived from these three.

        :param other: Another LatticeParameters object
        :type other: LatticeParameters

        :return: Equality boolean.
        :rtype: bool
        """
        return self.degree == other.degree and self.length == other.length and self.modulus == other.modulus

    def __repr__(self) -> str:
        """
        Output a canonical representation of the LatticeParameters object as a string.

        :return: Tuple containing degree, length, and modulus, cast as a string.
        :rtype: str
        """
        return str((self.degree, self.length, self.modulus))


UNIFORM_INFINITY_WEIGHT: str = 'inf,wt,unif'


def decode2coef_inf_unif(secpar: int, lp: LatticeParameters, val: str, btd: int,
                         dist_pars: Dict[str, int]) -> int:
    if btd < 1:
        raise ValueError('Cannot decode2coef_inf_unif without a positive integer number of bits required to decode.')
    elif 'bd' not in dist_pars or not isinstance(dist_pars['bd'], int) or not (1 <= dist_pars['bd'] <= lp.modulus // 2):
        raise ValueError('Cannot decode2coef_inf_unif without an integer bound 0 <= bd <= modulus//2.')
    elif btd < ceil(log2(dist_pars['bd'])) + 1 + secpar:
        b = dist_pars['bd']
        raise ValueError(
            f'Cannot decode2coef_inf_unif with secpar = {secpar}, bd = {b} without requiring ' +
            f'at least {ceil(log2(b)) + 1 + secpar} bits.'
        )
    elif not is_bitstring(val):
        raise ValueError('Cannot decode2coef_inf_unif without bitstring val.')
    elif len(val) < btd:
        raise ValueError(f'Cannot decode2coef_inf_unif without at least {btd} bits.')
    signum_bit: str = val[0]
    magnitude_minus_one_bits: str = val[1:]
    sign: int = 2 * int(signum_bit) - 1
    big_bd_flag = int(dist_pars['bd'] > 1)
    magnitude_minus_one: int = int(magnitude_minus_one_bits, 2)
    mag_minus_one_mod_bd: int = magnitude_minus_one % dist_pars['bd']
    magnitude: int = 1 + big_bd_flag * mag_minus_one_mod_bd
    return sign * magnitude


def decode2coef(secpar: int, lp: LatticeParameters, val: str, distribution: str, dist_pars: Dict[str, int],
                btd: int) -> int:
    """
    Decode an input string x to a coefficient in [-bd, -bd+1, ...,-2, -1, 1, 2, ..., bd-1, bd] with bias O(2**-secpar),
    if possible, and raise a ValueError if not possible. If bd = 1, this set is [-1, 1] and we only need 1 bit to
    sample from exactly the uniform distribution. On the other hand, if bd > 1, then we use the first bit of x as a sign
    bit, and we use the rest as the binary expansion of an integer. We mod this integer out by bd, add 1 to the result
    to get an integer in the set [1, 2, ..., bd], and then we multiply by +1 if the sign bit is 1 and -1 if the sign bit
    is 0 to get an integer in the set [-bd, -bd+1, ..., -2, -1, 1, 2, ..., bd - 1, bd].

    We require len(x) = ceil(log2(bd)) + 1 + secpar. This way, we have the ceil(log2(bd)) + secpar bits to determine the
    binary expansion of the integer, and an additional sign bit.

    The information-theoretic minimum of the number of bits required to uniquely determine an integer modulo bd is
    exactly ceil(log2(bd)). However, if x is a uniformly sampled ceil(log2(bd)) bit integer, then unless bd is a power
    of two, x % bd is not a uniformly distributed element of the integers modulo bd. If x is a uniformly
    sampled ceil(log2(bd))+k bit integer for some integer k, then the bias of x % bd away from the uniform distribution
    is O(2**-k). So to keep the bias negligible, we use secpar additional bits.

    :param secpar: Input security parameter
    :type secpar: int
    :param lp: Lattice parameters
    :type lp: LatticeParameters
    :param val: Input bitstring
    :type val: str
    :param distribution: String code indicating which distribution to use
    :type distribution: str
    :param dist_pars: Distribution parameters
    :type dist_pars: Dict[str, int]
    :param btd: Bits necessary to decode to an unbiased sample an integer modulo the modulus.
    :type btd: int

    :return: Return an integer uniformly selected from [-bd, 1-bd, ..., bd-1, bd] (or raise a ValueError).
    :rtype: int
    """
    if not is_bitstring(val):
        raise ValueError('Cannot decode2coef without a non-empty bitstring val.')
    elif not isinstance(distribution, str):
        raise ValueError('Cannot decode2coef without a string code indicating the distribution.')
    elif distribution == UNIFORM_INFINITY_WEIGHT:
        return decode2coef_inf_unif(secpar=secpar, lp=lp, val=val, dist_pars=dist_pars, btd=btd)
    raise ValueError('Tried to decode2coef with a distribution that has not yet been implemented.')


def decode2coefs(
        secpar: int, lp: LatticeParameters, distribution: str, dist_pars: Dict[str, int],
        val: str, num_coefs: int, btd: int
) -> List[int]:
    """
    Decode an input string x to a list of integer coefficients. In general, breaks the input string into blocks of
    1 + ceil(log2(bd)) + secpar bits each, and then merely calls decode2coef on each block. We do handle a weird edge
    case, when the bound is 1 (see decode2coef for more info on that). If bd == 1, we need wt bits, and otherwise we
    need wt * (ceil(log2(bd)) + 1 + secpar) bits.

    :param secpar: Security parameter
    :type secpar: int
    :param lp: Lattice parameters
    :type lp: LatticeParameters
    :param distribution: String code describing the distribution
    :type distribution: str
    :param dist_pars: Dictionary containing distribution parameters
    :type dist_pars: Dict[str, int]
    :param val: Input value/bitstring to decode.
    :type val: str
    :param num_coefs: Number of coefficients to sample.
    :type num_coefs: int
    :param btd: Number of bits required to decode a string to an unbiased sample of an integer modulo the modulus in lp.
    :type btd: int

    :return: Return a list of integers uniformly selected from [-bd, 1-bd, ..., bd-1, bd] (or raise a ValueError).
    :rtype: List[int]
    """
    if not isinstance(distribution, str):
        raise ValueError('Cannot decode2coefs without a string code describing the distribution.')
    elif num_coefs < 1:
        raise ValueError('Cannot decode2coefs without a number of coefficients to which we want to decode.')
    elif btd < 1:
        raise ValueError('Cannot decode2coefs without bits_to_decode, used to decode a single coefficient.')
    elif len(val) < num_coefs * btd:
        raise ValueError(f'Cannot decode2coefs without val with length at least {num_coefs * btd} but had {len(val)}')
    result = []
    for i in range(num_coefs):
        result += [decode2coef(
            secpar=secpar, lp=lp, distribution=distribution, dist_pars=dist_pars,
            btd=btd, val=val[i * btd: (i + 1) * btd]
        )]
    return result


def decode2indices(secpar: int, lp: LatticeParameters, num_coefs: int, val: str, bti: int) -> List[int]:
    """
    Decode input string x to a list of distinct, uniformly and independently sampled integer indices in [0, 1, ..., d-1]
    with constant weight equal to the input weight, wt, and with bias O(2**-secpar), if possible, and raise a ValueError
    if not possible. Requires ceil(log2(d)) + (wt - 1) * (ceil(log2(d)) + secpar) bits as input.

    First, all possible indices are stored into a possible_indices list, which is just list(range(d)).

    Next, the first ceil(log2(d)) bits of the input x are used directly to describe an integer modulo d with no bias,
    which we can call i for the purpose of this docstring. We pop possible_indices[i] out of the list (decreasing the
    length of the possible_indices list by 1) and store it in our result.

    Next, the remaining bits are split up into blocks of ceil(log2(d)) + secpar bits. Each block is cast as an integer
    with ceil(log2(d)) + secpar bits, and then modded out by len(possible_indices), resulting in another index i which
    has a distribution that is within O(2**-secpar) of uniform. We pop possible_indices[i] out again, and then move onto
    the next block until no blocks remain.

    :param secpar: Input security parameter
    :type secpar: int
    :param lp: Lattice parameters
    :type lp: LatticeParameters
    :param num_coefs: Number of coefficients to generate
    :type num_coefs: int
    :param val: Input bitstring
    :type val: str
    :param bti: Number of bits required to sample num_coefs worth of indices unbiasedly without replacement.
    :type bti: int

    :return: Return a list of length num_coefs, where each entry is a distinct integer in [0, 1, ..., d-1].
    :rtype: List[int]
    """
    if secpar < 1:
        raise ValueError('Cannot decode2indices without a positive integer security parameter.')
    elif not is_bitstring(val):
        raise ValueError('Cannot decode2indices without a bitstring val.')
    elif num_coefs < 1:
        raise ValueError('Cannot decode2indices with a sample size that is not a positive integer.')
    elif bti < ceil(log2(lp.degree)) + (num_coefs - 1) * (ceil(log2(lp.degree)) + secpar):
        a = ceil(log2(lp.degree))
        b = ceil(log2(lp.degree)) + secpar
        c = num_coefs - 1
        k = a + c * b
        raise ValueError(
            f'Cannot decode2indices without requiring at least {k} bits, but had {bti}.')
    elif len(val) < bti:
        raise ValueError(
            f'Cannot decode2indices without an input bitstring val with at least {bti} bits, ' +
            'but the input is only of length {len(val)}.'
        )
    possible_indices: List[int] = list(range(lp.degree))
    k: int = ceil(log2(len(possible_indices)))
    first_coef_bits: str = val[:k]
    first_coef: int = int(first_coef_bits, 2)
    result: list = list([possible_indices.pop(first_coef)])
    z: str = val[k:]
    k = ceil(log2(len(possible_indices)))
    j: int = k + secpar
    z: List[str] = [z[i * j: (i + 1) * j] for i in range(num_coefs - 1)]
    for next_z in z:
        next_z_as_int = int(next_z, 2)
        modded = next_z_as_int % len(possible_indices)
        result += [possible_indices.pop(modded)]
    return result


def decode2polycoefs(secpar: int, lp: LatticeParameters, distribution: str, dist_pars: Dict[str, int], val: str,
                     num_coefs: int, bti: int, btd: int) -> Dict[int, int]:
    """
    Decode an input string x to a dictionary with integer keys and values, suitable for use in creating a Polynomial
    object with norm bound bd and weight wt. We use the first ceil(log2(d)) + (wt-1) + (ceil(log2(d)) + secpar) bits to
    uniquely determine the index set, which we use decode2indices to compute. We use the rest of the bit string to
    determine the coefficients, which we use decode2coefs to compute. We always require at least
    ceil(log2(d)) + (wt - 1) * (ceil(log2(d)) + secpar) + wt bits, but when bd > 1, we also require an additional
    wt * (ceil(log2(bd)) + secpar) bits.

    :param secpar: Input security parameter
    :type secpar: int
    :param lp: Lattice parameters
    :type lp: LatticeParameters
    :param distribution: String code describing which distribution to use
    :type distribution: str
    :param dist_pars: Distribution parameters
    :type dist_pars: dict
    :param val: Input bitstring
    :type val: str
    :param num_coefs: Number of coefficients to generate
    :type num_coefs: int
    :param bti: Number of bits required to unbiasedly sample indices without replacement.
    :type bti: int
    :param btd: Number of bits required to unbiasedly sample an integer modulo the modulus in lp
    :type btd: int

    :return: Return a dict with integer keys and values, with wt distinct keys and all values in [-bd, ..., bd]
    :rtype: Dict[int, int]
    """
    if secpar < 1:
        raise ValueError('Cannot decode2polycoefs without an integer security parameter.')
    elif num_coefs < 1:
        raise ValueError('Cannot decode2polycoefs without an integer number of coefficients.')
    elif not is_bitstring(val):
        raise ValueError('Cannot decode2polycoefs without a bitstring val.')
    elif len(val) < 8 * get_gen_bytes_per_poly(
            secpar=secpar,
            lp=lp,
            distribution=distribution,
            dist_pars=dist_pars,
            num_coefs=dist_pars['wt'],
            btd=btd,
            bti=bti):
        raise ValueError('Cannot decode2polycoefs without a sufficiently long bitstring val.')
    elif bti != bits_to_indices(secpar=secpar, degree=lp.degree, wt=dist_pars['wt']):
        x = bits_to_indices(secpar=secpar, degree=lp.degree, wt=dist_pars['wt'])
        raise ValueError(f'Cannot decode2polycoefs without bits_to_ind == bits_to_indices, but had {bti} != {x}.')
    elif btd != bits_to_decode(secpar=secpar, bd=dist_pars['bd']):
        x = bits_to_decode(secpar=secpar, bd=dist_pars['bd'])
        raise ValueError(f'Cannot decode2polycoefs without bits_to_coef == bits_to_decode, but had {btd} != {x}.')
    x_for_indices: str = val[:bti]
    x_for_coefficients: str = val[bti:]
    indices: List[int] = decode2indices(
        secpar=secpar, lp=lp, bti=bti, num_coefs=num_coefs, val=x_for_indices
    )
    coefs: List[int] = decode2coefs(
        secpar=secpar, lp=lp, distribution=distribution, dist_pars=dist_pars,
        btd=btd, val=x_for_coefficients, num_coefs=num_coefs
    )
    return {index: coefficient for index, coefficient in zip(indices, coefs)}


def get_gen_bytes_per_poly_inf_wt_unif(
        secpar: int, lp: LatticeParameters, dist_pars: Dict[str, int], num_coefs: int
) -> int:
    if secpar < 1:
        raise ValueError('Cannot get_gen_bytes_per_poly_inf_wt_unif without an integer security parameter.')
    elif 'bd' not in dist_pars:
        raise ValueError('Cannot get_gen_bytes_per_poly_inf_wt_unif without a bound in dist_pars.')
    elif not isinstance(dist_pars['bd'], int):
        raise ValueError('Cannot get_gen_bytes_per_poly_inf_wt_unif without an integer bound in dist_pars')
    elif dist_pars['bd'] < 1 or dist_pars['bd'] > lp.modulus // 2:
        raise ValueError(
            'Cannot get_gen_bytes_per_poly_inf_wt_unif without an integer bound on [1, 2, ..., lp.modulus // 2].')
    elif 'wt' not in dist_pars:
        raise ValueError('Cannot get_gen_bytes_per_poly_inf_wt_unif without a weight in dist_pars.')
    elif not isinstance(dist_pars['wt'], int):
        raise ValueError('Cannot get_gen_bytes_per_poly_inf_wt_unif without an integer weight in dist_pars.')
    elif dist_pars['wt'] < 1 or dist_pars['wt'] > lp.degree:
        raise ValueError(
            'Cannot get_gen_bytes_per_poly_inf_wt_unif without an integer weight on [1, 2, .., lp.degree - 1].')
    elif dist_pars['wt'] != num_coefs:
        raise ValueError('Cannot get_gen_bytes_per_poly_inf_wt_unif with num_coefs != weight.')
    btd: int = bits_to_decode(secpar=secpar, bd=dist_pars['bd'])
    bti: int = bits_to_indices(secpar=secpar, degree=lp.degree, wt=num_coefs)
    return ceil((num_coefs * btd + bti)/8)


def get_gen_bytes_per_poly(secpar: int, lp: LatticeParameters, distribution: str, dist_pars: Dict[str, int],
                           num_coefs: int, bti: int, btd: int) -> int:
    """
    Compute bits required to decode a random bitstring to a polynomial of a certain weight and bound given some lattice
    parameters with a negligible bias away from uniformity. Note that this is not the same as the number of bits
    required to store a polynomial from the lattice! We require many more bits to sample, in order to ensure the
    distribution is negligibly different from uniform.

    Assumes a polynomial (which is a sum of monomials) is represented only by the coefficients and exponents of the
    nonzero summands. For example, to describe f(X) = 1 + 2 * X ** 7 + 5 * X ** 15, we can just store (0, 1), (7, 2),
    and (15, 5). If the polynomial has weight wt, then we store wt pairs. Each pair consists of an exponent from
    0, 1, ..., d-1, and each coefficient is an integer from [-(bd-1)//2, ..., (bd-1)//2]. In fact, to sample the first
    monomial exponent from [0, 1, ..., d - 1] with zero bias requires only ceil(log2(d)) bits, but sampling each
    subsequent monomial exponent requires an additional secpar bits. And, to sample the coefficient with negligible bias
    from uniform requires 1 + ceil(log2(bd)) + secpar bits.

    :param secpar: Input security parameter
    :type secpar: int
    :param lp: Lattice parameters
    :type lp: LatticeParameters
    :param distribution: String code describing which distribution to use
    :type distribution: str
    :param dist_pars: Distribution parameters
    :type dist_pars: dict
    :param num_coefs: Number of coefficients to generate
    :type num_coefs: int
    :param bti: Number of bits required to unbiasedly sample indices without replacement.
    :type bti: int
    :param btd: Number of bits required to unbiasedly sample an integer modulo the modulus in lp
    :type btd: int

    :return: Bits required to decode to a polynomial without bias.
    :rtype: int
    """
    if secpar < 1:
        raise ValueError('Cannot decode2polycoefs without an integer security parameter.')
    elif num_coefs < 1 or bti < 1 or btd < 1:
        raise ValueError(
            'Cannot decode2polycoefs without positive integer number of coefficients to generate and an ' +
            'integer number of indices to generate.'
        )
    elif distribution == UNIFORM_INFINITY_WEIGHT:
        return get_gen_bytes_per_poly_inf_wt_unif(secpar=secpar, lp=lp, dist_pars=dist_pars, num_coefs=num_coefs)
    raise ValueError(
        'We tried to compute the number of bits required to generate a polynomial for a distribution ' +
        'that is not supported.'
    )


class Polynomial(object):
    """
    Hybrid object with some container, some callable, and some numeric properties. Instantiated with a coefficient
    representation of a polynomial using the representation described in the comments of get_n_per_poly. On
    instantiation, the NTT value representation of the coefficient representation is computed and stored in self.vals
    and the coefficients are forgotten; all arithmetic is done with the NTT values.

    Attributes
    ----------
        lp: LatticeParameters
            For use in all arithmetic
        const_time_flag: bool
            Determines whether arithmetic should be done in constant time.
        ntt_representation: List[int]
            The NTT of the polynomial.

    Methods
    -------
        __init__(self)
            Initialize
        __eq__(self, other)
            Check for equality of polynomials
        __add__(self, other)
            Add two polynomials
        __radd__(self, other)
            Add two polynomials (or zero and a polynomial)
        __sub__(self, other)
            Subtract two polynomials
        __mul__(self, other)
            Multiply two polynomials
        __repr__(self)
            String representation of the polynomial
        reset_vals(self, coefs: Dict[int, int])
            Computes and stores ntt_values from coefs in self.vals, over-writing old value.
        get_coef_rep()
            Recompute and output the coefs from ntt_values, the infinity norm, and the weight
        to_bits()
            Convert to a bitstring
        to_bytes()
            Convert to a bytes object
    """
    lp: LatticeParameters
    const_time_flag: bool
    ntt_representation: List[int]

    def __init__(self, lp: LatticeParameters, coefs: Dict[int, int], const_time_flag: bool = True):
        """
        Initialize a polynomial object by passing in a LatticeParameters object and coefs, which is a Dict[int, int]
        where keys are monomial exponents and values are coefficients.

        :param lp: Input LatticeParameters
        :type lp: LatticeParameters
        :param coefs: Coefficient dictionary
        :type coefs: Dict[int, int]
        """
        if len(coefs) > lp.degree:
            raise ValueError('Cannot create polynomial with too many coefficients specified.')
        elif not all(0 <= i < lp.degree for i in coefs):
            raise ValueError(
                f'Cannot create a polynomial with monomial exponents outside of [0, 1, ..., {lp.degree - 1}]'
            )
        self.lp = lp
        self.const_time_flag = const_time_flag
        self._reset_vals(coefs=coefs)  # set the ntt_representation

    def __eq__(self, other) -> bool:
        """
        Check if self and other have the same lattice parameters and generate the same coefficients.

        :param other: Another Polynomial.
        :type other: Polynomial

        :return: Boolean indicating Polynomial equality
        :rtype: bool
        """
        if self.lp != other.lp:
            return False
        x = self.get_coef_rep()
        y = other.get_coef_rep()
        return x == y

    def __add__(self, other):
        """
        Add two polynomials.

        :param other: Another Polynomial
        :type other: Polynomial
        :return: Sum of self and other.
        :rtype: Polynomial
        """
        if isinstance(other, int) and other == 0:
            return self
        result = deepcopy(self)
        if self.const_time_flag:
            result.ntt_representation = [
                cent(q=self.lp.modulus, halfmod=self.lp.halfmod, logmod=self.lp.logmod, val=x + y)
                for x, y in zip(result.ntt_representation, other.ntt_representation)
            ]
        else:
            result.ntt_representation = [
                (x + y) % self.lp.modulus for x, y in zip(result.ntt_representation, other.ntt_representation)
            ]
            result.ntt_representation = [
                x if x <= self.lp.modulus // 2 else x - self.lp.modulus for x in result.ntt_representation
            ]
        return result

    def __radd__(self, other):
        """
        Add two polynomials.

        :param other: Another Polynomial
        :type other: Polynomial
        :return: Sum of self and other.
        :rtype: Polynomial
        """
        if isinstance(other, int) and other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other):
        """
        Subtract two polynomials.

        :param other: Another Polynomial
        :type other: Polynomial
        :return: Sum of self and other.
        :rtype: Polynomial
        """
        if isinstance(other, int) and other == 0:
            return self
        result = deepcopy(self)
        if self.const_time_flag:
            result.ntt_representation = [
                cent(q=self.lp.modulus, halfmod=self.lp.halfmod, logmod=self.lp.logmod, val=x - y)
                for x, y in zip(result.ntt_representation, other.ntt_representation)
            ]
        else:
            result.ntt_representation = [
                (x - y) % self.lp.modulus for x, y in zip(result.ntt_representation, other.ntt_representation)
            ]
            result.ntt_representation = [
                x if x <= self.lp.modulus // 2 else x - self.lp.modulus for x in result.ntt_representation
            ]
        return result

    def __mul__(self, other):
        """
        Multiply two polynomials

        :param other: Other Polynomial
        :type other: Polynomial
        :return:
        :rtype: Polynomial
        """
        if isinstance(other, int) and other == 0:
            return 0
        result = deepcopy(self)
        if self.const_time_flag:
            result.ntt_representation = [
                cent(q=self.lp.modulus, halfmod=self.lp.halfmod, logmod=self.lp.logmod, val=x * y)
                for x, y in zip(result.ntt_representation, other.ntt_representation)
            ]
        else:
            result.ntt_representation = [
                (x * y) % self.lp.modulus for x, y in zip(result.ntt_representation, other.ntt_representation)
            ]
            result.ntt_representation = [
                x if x <= self.lp.modulus // 2 else x - self.lp.modulus for x in result.ntt_representation
            ]
        return result

    def __rmul__(self, other):
        """ Use NTT representation to point-wise multiply polynomial ntt_values.

        :param other: Other Polynomial
        :type other: Polynomial
        :return:
        :rtype: Polynomial
        """
        if isinstance(other, int) and other == 0:
            return 0
        return self.__mul__(other)

    def __repr__(self) -> str:
        """
        Return a canonical string representation of the Polynomial; WARNING: calls get_coefs.

        :return:
        :rtype: str
        """
        coef_rep, norm, wt = self.get_coef_rep()
        sorted_keys = sorted(list(coef_rep.keys()))
        sorted_coefs = [(i, coef_rep[i]) for i in sorted_keys]
        return str((sorted_coefs, norm, wt))

    def _ntt(self, inv_flag: bool, val: List[int]) -> List[int]:
        """
        Very thin wrapper that attaches ntt method to the Polynomial object

        :param inv_flag: Indicates whether we are performing forward NTT or inverse NTT
        :type inv_flag: bool
        :return: Return the NTT (or inverse) of the inputs x.
        :rtype: List[int]
        """
        return ntt(
            q=self.lp.modulus, zetas=self.lp.zetas, zetas_inv=self.lp.zetas_invs, val=val,
            halfmod=self.lp.halfmod, logmod=self.lp.logmod, n=self.lp.n, lgn=self.lp.lgn, inv_flag=inv_flag,
            const_time_flag=self.const_time_flag
        )

    def _reset_vals(self, coefs: Dict[int, int]) -> None:
        """ Input a Dict[int, int] of input coefficients (keys are exponents, values are coefficients), compute the NTT
        of the result, then overwrite self.vals with the new NTT data.

        :param coefs: Input coefficients
        :type coefs: Dict[int, int]
        """
        tmp: List[int] = [0 for _ in range(self.lp.n)]
        for i in range(self.lp.n):
            if i in coefs:
                tmp[i] += coefs[i]
        self.ntt_representation = self._ntt(inv_flag=False, val=tmp)

    def get_coef_rep(self) -> Tuple[Dict[int, int], int, int]:
        """
        Compute the coefficient representation of the polynomial by performing the inverse NTT on self.vals, compute the
        norm and the wight, and return all these.

        :return: Coefficient representation of the Polynomial, norm, and weight.
        :rtype: Tuple[Dict[int, int], int, int]
        """
        tmp: List[int] = self._ntt(inv_flag=True, val=self.ntt_representation)
        left: List[int] = tmp[:self.lp.degree]
        right: List[int] = tmp[self.lp.degree:]
        if self.const_time_flag:
            coefs: List[int] = [
                cent(q=self.lp.modulus, halfmod=self.lp.halfmod, logmod=self.lp.logmod, val=x - y)
                for x, y in zip(left, right)
            ]
        else:
            coefs: List[int] = [(x - y) % self.lp.modulus for x, y in zip(left, right)]
            coefs = [x if x <= self.lp.modulus // 2 else x - self.lp.modulus for x in coefs]
        coefs_dict: Dict[int, int] = {index: value for index, value in enumerate(coefs) if value != 0}
        if not coefs_dict:
            return coefs_dict, 0, 0
        return coefs_dict, max(abs(coefs_dict[value]) for value in coefs_dict), len(coefs_dict)

    def to_bytes(self) -> bytearray:
        return bytearray(self.ntt_representation)

    def to_bits(self) -> str:
        return sum(bin(i % self.lp.modulus)[2:] for i in self.ntt_representation)


def decode2poly(
        secpar: int, lp: LatticeParameters, distribution: str, dist_pars: Dict[str, int], val: str,
        num_coefs: int, bti: int, btd: int, const_time_flag: bool = True
) -> Polynomial:
    return Polynomial(
        lp=lp,
        coefs=decode2polycoefs(secpar=secpar, lp=lp, distribution=distribution, dist_pars=dist_pars, val=val,
                               num_coefs=num_coefs, bti=bti, btd=btd),
        const_time_flag=const_time_flag,
    )


def hash2polynomial(secpar: int, lp: LatticeParameters, distribution: str, dist_pars: Dict[str, int], salt: str,
                    msg: str, num_coefs: int, bti: int, btd: int,
                    const_time_flag: bool = True) -> Polynomial:
    """
    Hash an input message msg and salt to a polynomial with norm bound at most bd and weight at most wt.

    :param secpar: Input security parameter
    :type secpar: int
    :param lp: Lattice parameters
    :type lp: LatticeParameters
    :param distribution: String code describing which distribution to use
    :type distribution: str
    :param dist_pars: Distribution parameters
    :type dist_pars: dict
    :param salt: Salt
    :type salt: str
    :param msg: Message being hashed
    :type msg: str
    :param num_coefs: Number of coefficients to generate
    :type num_coefs: int
    :param bti: Number of bits required to unbiasedly sample indices without replacement.
    :type bti: int
    :param btd: Number of bits required to unbiasedly sample an integer modulo the modulus in lp
    :type btd: int
    :param const_time_flag: Boolean indicating whether arithmetic should be const-time.
    :type const_time_flag: bool

    :return:
    :rtype: Polynomial
    """
    num_bytes_for_hashing: int = get_gen_bytes_per_poly(
        secpar=secpar, lp=lp, distribution=distribution, dist_pars=dist_pars, num_coefs=num_coefs, bti=bti, btd=btd
    )
    val: str = binary_digest(msg, num_bytes_for_hashing, salt)
    coefs: Dict[int, int] = decode2polycoefs(secpar=secpar, lp=lp, distribution=distribution, dist_pars=dist_pars,
                                             val=val, num_coefs=num_coefs, bti=bti, btd=btd)
    return Polynomial(lp=lp, coefs=coefs, const_time_flag=const_time_flag)


def random_polynomial(
        secpar: int, lp: LatticeParameters, distribution: str, dist_pars: Dict[str, int], num_coefs: int,
        bti: int, btd: int, const_time_flag: bool = True
) -> Polynomial:
    """
    Generate a random polynomial with norm bounded by bd and weight bounded by wt. Relies on randbits from
    the secrets package to generate data. Since the secrets package is thought to be secure for cryptographic use, the
    results should have a negligible bias away from uniformity.

    :param secpar: Input security parameter
    :type secpar: int
    :param lp: Lattice parameters
    :type lp: LatticeParameters
    :param distribution: String code describing which distribution to use
    :type distribution: str
    :param dist_pars: Distribution parameters
    :type dist_pars: dict
    :param num_coefs: Number of coefficients to generate
    :type num_coefs: int
    :param bti: Number of bits required to unbiasedly sample indices without replacement.
    :type bti: int
    :param btd: Number of bits required to decode to an unbiased sample an integer modulo the modulus in lp
    :type btd: int
    :param const_time_flag: Indicates whether arithmetic should be constant time.
    :type const_time_flag: bool

    :return:
    :rtype: Polynomial
    """
    num_bits_for_hashing: int = 8 * get_gen_bytes_per_poly(
        secpar=secpar,
        lp=lp,
        distribution=distribution,
        dist_pars=dist_pars,
        num_coefs=num_coefs,
        btd=btd,
        bti=bti)
    val: str = bin(randbits(num_bits_for_hashing))[2:].zfill(num_bits_for_hashing)

    return decode2poly(
        secpar=secpar, lp=lp, distribution=distribution, dist_pars=dist_pars, val=val, num_coefs=num_coefs,
        bti=bti, btd=btd, const_time_flag=const_time_flag
    )


class PolynomialVector(object):
    """
    Contains LatticeParameters and a list of polynomials. WARNING: We repurpose the pow notation for scaling by a poly.

    Attributes
    ----------
        lp: LatticeParameters
            For use in all arithmetic.
        const_time_flag: bool
            Indicates whether arithmetic should be constant time.
        entries: List[Polynomial]
            The "vector" of polynomials.

    Methods
    -------
        __init__(self)
            Initialize
        __eq__(self, other)
            Check for equality of PolynomialVectors
        __add__(self, other)
            Add two PolynomialVectors
        __radd__(self, other)
            Add two PolynomialVectors (or zero and a PolynomialVectors)
        __sub__(self, other)
            Subtract two PolynomialVectors
        __mul__(self, other)
            Compute the dot product of two polynomial vectors.
        __pow__(self, other: Polynomial)
            Scale self
        __repr__(self)
            String representation of the polynomial
        norm_and_weight()
            Return a tuple with both norm and weight.
        norm()
            Return the maximum of the absolute value of the values in coefs
        weight()
            Return the length of coefs.
    """
    lp: LatticeParameters
    const_time_flag: bool
    entries: List[Polynomial]

    def __init__(self, lp: LatticeParameters, entries: List[Polynomial], const_time_flag: bool = True):
        """
        Instantiate with some input LatticeParameters and a list of Polynomial entries.

        :param lp: Input lattice parameters
        :type lp: LatticeParameters
        :param entries: Input polynomial entries
        :type entries: List[Polynomial]
        """
        if not all(i.lp == lp for i in entries):
            raise ValueError('Can only create PolynomialVector with all common lattice parameters.')
        elif not all(i.const_time_flag == const_time_flag for i in entries):
            raise ValueError('The const_time_flag for each entry in the PolynomialVector must match ' +
                             'the const_time_flag of the PolynomialVector.')
        self.lp = lp
        self.const_time_flag = const_time_flag
        self.entries = entries

    def __eq__(self, other) -> bool:
        """
        PolynomialVector equality is determined if parameters match and all entries match.

        :param other: Other PolynomialVector
        :type other: PolynomialVector
        :return: Boolean indicating whether parameters and all entries match
        :rtype: bool
        """
        return self.lp == other.lp and self.entries == other.entries

    def __add__(self, other):
        """
        Add PolynomialVectors

        :param other: Other PolynomialVector
        :type other: PolynomialVector
        :return: self + other
        :rtype: PolynomialVector
        """
        result = deepcopy(self)
        result.entries = [x + y for x, y in zip(result.entries, other.entries)]
        return result

    def __radd__(self, other):
        """
        Add PolynomialVectors

        :param other: Other PolynomialVector
        :type other: PolynomialVector
        :return: self + other
        :rtype: PolynomialVector
        """
        if other == 0:
            return self
        return self.__add__(other)

    def __sub__(self, other):
        """
        Add PolynomialVectors

        :param other: Other PolynomialVector
        :type other: PolynomialVector
        :return: self - other
        :rtype: PolynomialVector
        """
        result = deepcopy(self)
        result.entries = [x - y for x, y in zip(result.entries, other.entries)]
        return result

    def __mul__(self, other) -> Polynomial:
        """
        Dot product between two polynomial vectors.

        :param other: Other PolynomialVector
        :type other: PolynomialVector

        :return: The dot product
        :rtype: PolynomialVector
        """
        if self.lp != other.lp:
            raise ValueError('Can only compute dot products with polynomials with the same parameters.')
        each_product = [x * y for x, y in zip(self.entries, other.entries)]
        return sum(each_product)

    def __pow__(self, scalar: Polynomial):
        """
        Scale a PolynomialVector by a Polynomial scalar. So if x is a PolynomialVector and y is a Polynomial, z = x ** y
        gives us the PolynomialVector such that the ith entry is z.entries[i] == x.entries[i] * y.

        :param scalar: Input polynomial for scaling
        :type scalar: Polynomial
        :return:
        :rtype: PolynomialVector
        """
        # Abuse ** operator to scale vectors: cv = v**c
        if scalar.const_time_flag != self.const_time_flag:
            raise ValueError('Cannot scale a PolynomialVector with a different const_time_flag than the scalar.')
        result = deepcopy(self)
        result.entries = [scalar * i for i in result.entries]
        return result

    def __repr__(self) -> str:
        """
        A canonical string representation of a PolynomialVector object: str(self.entries).

        :return:
        :rtype: str
        """
        return str(self.entries)

    def get_coef_rep(self) -> List[Tuple[Dict[int, int], int, int]]:
        """
        Calls get_coef_rep for each entry.

        :return:
        :rtype: List[Tuple[Dict[int, int], int, int]]
        """
        return [val.get_coef_rep() for i, val in enumerate(self.entries)]

    def to_bytes(self) -> bytearray:
        return sum(i.to_bytes() for i in self.entries)

    def to_bits(self) -> str:
        return sum(i.to_bits() for i in self.entries)


def decode2polynomialvector(
        secpar: int, lp: LatticeParameters, distribution: str, dist_pars: Dict[str, int], val: str,
        num_coefs: int, bti: int, btd: int, const_time_flag: bool = True
) -> PolynomialVector:
    if not is_bitstring(val):
        raise ValueError('Can only decode to a polynomial vector with an input bitstring val.')
    k: int = 8 * get_gen_bytes_per_poly(
        secpar=secpar, lp=lp, distribution=distribution, dist_pars=dist_pars, num_coefs=num_coefs,
        bti=bti, btd=btd
    )
    if len(val) < k * lp.length:
        raise ValueError(
            f'Cannot decode2polynomialvector without an input bitstring val with length at ' +
            f'least {k} bits, but had length {len(val)}.'
        )
    entries = [
        Polynomial(
            lp=lp,
            coefs=decode2polycoefs(secpar=secpar, lp=lp, distribution=distribution, dist_pars=dist_pars,
                                   val=val[i * k: (i + 1) * k], num_coefs=num_coefs, bti=bti, btd=btd),
            const_time_flag=const_time_flag,
        ) for i in range(lp.length)
    ]
    return PolynomialVector(lp=lp, entries=entries, const_time_flag=const_time_flag)


def random_polynomial_vector_inf_wt_unif(
        secpar: int, lp: LatticeParameters, dist_pars: Dict[str, int], num_coefs: int,
        bti: int, btd: int, const_time_flag: bool = True
) -> PolynomialVector:
    if 'bd' not in dist_pars or not isinstance(dist_pars['bd'], int) or \
            dist_pars['bd'] < 1 or dist_pars['bd'] > lp.modulus // 2:
        raise ValueError(
            'Cannot random_polynomial_vector_inf_wt_unif without positive integer bound less than half the modulus.'
        )
    elif 'wt' not in dist_pars or not isinstance(dist_pars['wt'], int) or \
            dist_pars['wt'] < 1 or dist_pars['wt'] > lp.degree:
        raise ValueError('Cannot random_polynomial_vector_inf_wt_unif without positive integer weight.')
    k = 8 * lp.length * get_gen_bytes_per_poly(
        secpar=secpar, lp=lp, distribution=UNIFORM_INFINITY_WEIGHT, dist_pars=dist_pars,
        num_coefs=num_coefs, bti=bti, btd=btd
    )
    return decode2polynomialvector(
        secpar=secpar, lp=lp, distribution=UNIFORM_INFINITY_WEIGHT, dist_pars=dist_pars,
        num_coefs=num_coefs, bti=bti, btd=btd, val=bin(randbits(k))[2:].zfill(k), const_time_flag=const_time_flag
    )


def random_polynomialvector(
        secpar: int, lp: LatticeParameters, distribution: str, dist_pars: Dict[str, int], num_coefs: int,
        bti: int, btd: int, const_time_flag: bool = True
) -> PolynomialVector:
    """
    Generate a random PolynomialVector with bounded Polynomial entries. Essentially just instantiates a PolynomialVector
    object with a list of random Polynomial objects as entries, which are in turn generated by random_polynomial

    :param secpar: Input security parameter
    :type secpar: int
    :param lp: Lattice parameters
    :type lp: LatticeParameters
    :param distribution: String code describing which distribution to use
    :type distribution: str
    :param dist_pars: Distribution parameters
    :type dist_pars: dict
    :param num_coefs: Number of coefficients to generate
    :type num_coefs: int
    :param bti: Number of bits required to unbiasedly sample indices without replacement.
    :type bti: int
    :param btd: Number of bits required to unbiasedly sample an integer modulo the modulus in lp
    :type btd: int
    :param const_time_flag: Indicates whether arithmetic should be constant time.
    :type const_time_flag: bool

    :return:
    :rtype: PolynomialVector
    """
    if secpar < 1:
        raise ValueError('Cannot random_polynomialvector without an integer security parameter.')
    elif distribution == UNIFORM_INFINITY_WEIGHT:
        return random_polynomial_vector_inf_wt_unif(
            secpar=secpar, lp=lp, dist_pars=dist_pars, num_coefs=num_coefs,
            bti=bti, btd=btd, const_time_flag=const_time_flag
        )
    raise ValueError('Tried to random_polynomialvector with a distribution that is not supported.')


def hash2polynomialvector(
        secpar: int, lp: LatticeParameters, distribution: str, dist_pars: Dict[str, int], num_coefs: int,
        bti: int, btd: int, msg: str, salt: str, const_time_flag: bool = True
) -> PolynomialVector:
    """
    Hash an input message msg and salt to a polynomial vector with norm bound at most bd and weight at most wt. Just
    calls decode2polycoefs repeatedly.

    :param secpar: Input security parameter
    :type secpar: int
    :param lp: Lattice parameters
    :type lp: LatticeParameters
    :param distribution: String code describing which distribution to use
    :type distribution: str
    :param dist_pars: Distribution parameters
    :type dist_pars: dict
    :param salt: Salt
    :type salt: str
    :param msg: Message being hashed
    :type msg: str
    :param num_coefs: Number of coefficients to generate
    :type num_coefs: int
    :param bti: Number of bits required to unbiasedly sample indices without replacement.
    :type bti: int
    :param btd: Number of bits required to unbiasedly sample an integer modulo the modulus in lp
    :type btd: int
    :param const_time_flag: Indicates whether arithmetic should be constant time.
    :type const_time_flag: bool

    :return: Call decode2polycoefs for length, create Polynomial for each, return a PolynomialVector with these entries
    :rtype: PolynomialVector
    """
    k: int = lp.length * get_gen_bytes_per_poly(
        secpar=secpar, lp=lp, distribution=distribution, dist_pars=dist_pars, num_coefs=num_coefs, bti=bti, btd=btd
    )
    val: str = binary_digest(msg=msg, num_bytes=k, salt=salt)
    return decode2polynomialvector(
        secpar=secpar, lp=lp, distribution=distribution, dist_pars=dist_pars, val=val,
        num_coefs=num_coefs, bti=bti, btd=btd, const_time_flag=const_time_flag
    )
