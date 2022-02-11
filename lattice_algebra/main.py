"""
The lattice_algebra module is an (unaudited) prototype containing basic algebraic infrastructure for employing lattice-
based crypto schemes in the Ring (aka Module or Ideal) Short Integer Solution and Ring (Module, Ideal) Learning With
Errors settings where secret/error vectors are a uniformly distributed from the subset of vectors with bounded infinity
norm and constant weight. 

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
from secrets import randbits, randbelow
from hashlib import shake_256 as shake
from typing import List, Dict, Tuple


def is_prime(q: int) -> bool:
    """
    Test whether input integer is prime with the Sieve of Eratosthenes.

    :param q: Input value
    :type q: int

    :return: Indicate whether q is prime.
    :rtype: bool
    """
    return all([q % i != 0 for i in range(2, ceil(sqrt(q)) + 1)])


def is_pow_two(val: int) -> bool:
    """
    Test whether input integer is a power of two by summing the bits in its binary expansion and testing for 1.

    :param val: Input value
    :type val: int

    :return: Indicate whether x is a power-of-two.
    :rtype: bool
    """
    return val > 0 and not (val & (val - 1))


def is_ntt_friendly_prime(q: int, d: int) -> bool:
    """
    Test whether input integer pair can be used to construct an NTT-friendly ring.

    :param q: Input modulus
    :type q: int
    :param d: Input degree
    :type d: int

    :return: Indicate whether q is prime and q-1 == 0 mod 2d.
    :rtype: bool
    """
    return is_prime(q) and is_pow_two(d) and q % (2 * d) == 1


def is_prim_rou(q: int, d: int, val: int) -> bool:
    """
    Test whether input x is a primitive 2d^th root of unity in the integers modulo q. Does not require q and d to be
    an ntt-friendly pair.

    :param q: Input modulus
    :type q: int
    :param d: Input degree
    :type d: int
    :param val: Purported root of unity
    :type val: int

    :return: Boolean indicating x**(2d) == 1 and x**i != 1 for 1 <= i < 2d.
    :rtype: bool
    """
    return all(val ** k % q != 1 for k in range(1, 2 * d)) and val ** (2 * d) % q == 1


def get_prim_rou_and_rou_inv(q: int, d: int) -> Tuple[int, int]:
    """
    Compute a primitive 2d-th root of unity modulo q and its inverse. Raises a ValueError if (d, q) are not an ntt-
    friendly pair. Works by finding the first (in natural number order) primitive root of unity and its inverse.

    :param q: Input modulus
    :type q: int
    :param d: Input degree
    :type d: int

    :return: Root of unity and its inverse in a tuple.
    :rtype: Tuple[int, int]
    """
    if not (is_ntt_friendly_prime(q, d)):
        raise ValueError('Input q and d are not ntt-friendly prime and degree.')
    x: int = 2
    while x < q:
        if is_prim_rou(q, d, x):
            break
        x += 1
    return x, ((x ** (2 * d - 1)) % q)


def is_bitstring(val: str) -> bool:
    """
    Ensure that an ostensible bitstring does not contain any characters besides 0 and 1 (without type conversion)

    :param val: Input to to be validated as a bitstring
    :type val: str

    :return: True if the input is a bitstring (i.e. contains only 0's and 1's) and False otherwise.
    :rtype: bool
    """
    return ''.join(sorted(list(set(val)))) in '01'


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
    # TODO: pre-compute bit_reverse for each n-bit x or use amortized reverse binary counter scheme from 17-1 in Rivest
    #  et al's "Introduction to Algorithms"
    if (num_bits, val) not in touched_bit_rev:
        x_in_bin: str = bin(val)[2:].zfill(num_bits)
        touched_bit_rev[(num_bits, val)] = int(x_in_bin[::-1], 2)
    return touched_bit_rev[(num_bits, val)]


def bit_rev_cp(val: List[int]) -> List[int]:
    """
    Permute the indices of the input list x to bit-reversed order. Note: does not compute the bit-reverse of the
    values in the input list, just shuffles the input list around.
    TODO: Input n instead of recomputing it every time.

    :param val: Input values
    :type val: List[int]

    :return: Output permuted list
    :rtype: List[int]
    """
    if not is_pow_two(len(val)):
        raise ValueError("Can only bit-reverse-copy arrays with power-of-two lengths.")
    n: int = ceil(log2(len(val)))
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


def make_zetas_and_invs(q: int, d: int, halfmod: int, logmod: int, n: int, lgn: int) -> Tuple[List[int], List[int]]:
    """
    Compute powers of primitive root of unity and its inverse for use in the NTT function. Finds the root of unity,
    say zeta, and its inverse, say zeta_inv, then outputs [zeta ** ((2d) // (2**(s+1))) for s in range(log2(2d))]
    and [zeta_inv ** ((2d) // (2**(s+1))) for s in range(log2(2d))], modulo q.

    :param q: Input modulus
    :type q: int
    :param d: Input degree
    :type d: int
    :param halfmod: q//2
    :type halfmod: int
    :param logmod: ceil(log2(q))
    :type logmod: int
    :param n: 2*d
    :type n: int
    :param lgn: ceil(log2(n))
    :type lgn: int

    :return: Return the powers of zeta and zeta_inv for use in NTT, two lists of integers, in a tuple
    :rtype: Tuple[List[int], List[int]]
    """
    powers: List[int] = [n // (2 ** (s + 1)) for s in range(lgn)]
    zeta, zeta_inv = get_prim_rou_and_rou_inv(q, d)
    left: List[int] = [cent(q=q, halfmod=halfmod, logmod=logmod, val=int(zeta ** i)) for i in powers]
    right: List[int] = [cent(q, halfmod=halfmod, logmod=logmod, val=int(zeta_inv ** i)) for i in powers]
    return left, right


def ntt(q: int, zetas: List[int], zetas_inv: List[int], inv_flag: bool, halfmod: int, logmod: int, n: int, lgn: int,
        val: List[int]) -> List[int]:
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


    :return: Return the NTT (or INTT) of the inputs x.
    :rtype: List[int]
    """
    if sum(int(i) for i in bin(len(val))[2:]) != 1:
        raise ValueError("Can only NTT arrays with lengths that are powers of two.")
    bit_rev_x: List[int] = bit_rev_cp(val)
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
                bit_rev_x[k + j]: int = cent(q=q, halfmod=halfmod, logmod=logmod, val=u + t)
                bit_rev_x[k + j + m // 2]: int = cent(q=q, halfmod=halfmod, logmod=logmod, val=u - t)
                w *= this_zeta
    if inv_flag:
        n_inv: int = 1
        while (n_inv * n) % q != 1:
            n_inv += 1
        bit_rev_x: List[int] = [cent(q=q, halfmod=halfmod, logmod=logmod, val=(n_inv * i)) for i in bit_rev_x]
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


def decode2coef(secpar: int, bd: int, val: str) -> int:
    """
    Decode an input string x to a coefficient in [-bd, -bd+1, ...,-2, -1, 1, 2, ..., bd-1, bd] with bias
    O(2**-secpar), if possible, and raise a ValueError if not possible. If bd = 1, this set is [-1, 1] and we only need
    one bit to sample from exactly the uniform distribution. On the other hand, if bd > 1, then we use the first bit of
    x as a sign bit, and we use the rest as the binary expansion of an integer. We mod this integer out by bd, add 1 to
    the result to get an integer in the set [1, 2, ..., bd], and then we multiply by +1 if the sign bit is 1 and -1 if
    the sign bit is 0 to get an integer in the set [-bd, -bd+1, ..., -2, -1, 1, 2, ..., bd - 1, bd].

    We require len(x) = ceil(log2(bd)) + 1 + secpar. This way, we have the ceil(log2(bd)) + secpar bits to determine the
    binary expansion of the integer, and an additional sign bit.

    The information-theoretic minimum of the number of bits required to uniquely determine an integer modulo bd is
    exactly ceil(log2(bd)). However, if x is a uniformly sampled ceil(log2(bd)) bit integer, then unless bd is a power
    of two, x % bd is not a uniformly distributed element of the integers modulo bd. If x is a uniformly sampled
    ceil(log2(bd))+k bit integer for some integer k, then the bias of x % bd away from the uniform distribution is
    O(2**-k). So to keep the bias negligible, we use secpar additional bits.

    :param secpar: Input security parameter
    :type secpar: int
    :param bd: Input bound
    :type bd: int
    :param val: Input bitstring
    :type val: str

    :return: Return an integer uniformly selected from [-bd, 1-bd, ..., bd-1, bd] (or raise a ValueError).
    :rtype: int
    """
    if not val:
        raise ValueError('Cannot decode an empty bitstring.')
    elif not all(int(i) in [0, 1] for i in val):
        raise ValueError('Cannot decode a polynomial coefficient from a non-bitstring.')
    elif bd < 1:
        raise ValueError('Cannot generate a non-zero coefficient between 0 and 0.')
    elif bd == 1:
        # In this case, we are sampling from [-1, 1] uniformly, so we only really need one bit for uniformity.
        return 2 * int(val[0]) - 1
    elif len(val) < ceil(log2(bd)) + 1 + secpar:
        raise ValueError('Bitstring not long enough to encode a bounded coefficient.')
    return (2 * int(val[0]) - 1) * (1 + (int(val[1:], 2) % bd))


def decode2coefs(secpar: int, bd: int, wt: int, val: str) -> List[int]:
    """
    Decode an input string x to a list of integer coefficients. In general, breaks the input string into blocks of
    1 + ceil(log2(bd)) + secpar bits each, and then merely calls decode2coef on each block. We do handle one weird edge
    case, when the bound is 1 (see decode2coef for more info on that).

    If bd == 1, we need wt bits, and otherwise we need wt * (ceil(log2(bd)) + 1 + secpar) bits.

    TODO: Can this be made more pythonic?

    :param secpar: Input security parameter
    :type secpar: int
    :param bd: Input bound
    :type bd: int
    :param wt: Input weight
    :type wt: int
    :param val: Input bitstring
    :type val: str

    :return: Return a list of integers uniformly selected from [-bd, 1-bd, ..., bd-1, bd] (or raise a ValueError).
    :rtype: List[int]
    """
    if (bd == 1 and len(val) < wt) or bd > 1 and len(val) < wt * (
            ceil(log2(bd)) + 1 + secpar):
        raise ValueError('Bitstring not long enough to encode all the coefficients.')
    elif not is_bitstring(val):
        raise ValueError(f'Cannot decode polynomial coefficients from a non-bitstring. Problem: {val}')
    tmp: int = 1
    if bd > 1:
        tmp += ceil(log2(bd)) + secpar
    y: List[str] = [val[i * tmp: (i + 1) * tmp] for i in range(wt)]
    return [decode2coef(secpar, bd, i) for i in y]


def decode2indices(secpar: int, d: int, wt: int, val: str) -> List[int]:
    """
    Decode an input string x to a list of distinct integer indices in [0, 1, ..., d-1] with constant weight
    equal to the input weight, wt, and with bias O(2**-secpar), if possible, and raise a ValueError if not possible. Re-
    quires ceil(log2(d)) + (wt - 1) * (ceil(log2(d)) + secpar) bits as input.

    First, all possible indices are stored into a possible_indices list, which is just list(range(d)).

    Next, the first ceil(log2(d)) bits of the input x are used directly to describe an integer modulo d with no
    bias, which we can call i for the purpose of this docstring. We pop possible_indices[i] out of the list (decreasing
    the length of the possible_indices list by 1) and store it in our result.

    Next, the remaining bits are split up into blocks of ceil(log2(d)) + secpar bits. Each block is cast as an integer
    with ceil(log2(d)) + secpar bits, and then modded out by len(possible_indices), resulting in another index
    i which has a distribution that is within O(2**-secpar) of uniform. We pop possible_indices[i] out again, and then
    move onto the next block until no blocks remain.

    :param secpar: Input security parameter
    :type secpar: int
    :param d: Input degree
    :type d: int
    :param wt: Input weight
    :type wt: int
    :param val: Input bitstring
    :type val: str

    :return: Return a list of length wt, where each entry is a distinct integer in [0, 1, ..., d-1].
    :rtype: List[int]
    """
    if len(val) < ceil(log2(d)) + (wt - 1) * (ceil(log2(d)) + secpar):
        raise ValueError('Bitstring not long enough to encode an indicator vector of length degree = ' + str(
            d) + ' with weight = ' + str(wt))  # fstring??
    elif not is_bitstring(val):
        raise ValueError('Cannot decode polynomial coefficient indices from a non-bitstring.')
    possible_indices: List[int] = list(range(d))
    result: list = list([possible_indices.pop(int(val[:ceil(log2(d))], 2) % d)])
    z: str = val[ceil(log2(d)):]
    z: List[str] = [z[i * (ceil(log2(d)) + secpar): (i + 1) * (ceil(log2(d)) + secpar)] for i in range(wt - 1)]
    for next_z in z:
        result += [possible_indices.pop(int(next_z, 2) % len(possible_indices))]
    return result


def decode2polycoefs(secpar: int, d: int, bd: int, wt: int, val: str) -> Dict[int, int]:
    """
    Decode an input string x to a dictionary with integer keys and values, suitable for use in creating a Polynomial
    object with norm bound bd and weight wt. We use the first ceil(log2(d)) + (wt-1) + (ceil(log2(d)) + secpar) bits
    to uniquely determine the index set, which we use decode2indices to compute. We use the rest of the bit string
    to determine the coefficients, which we use decode2coefs to compute.

    We always require at least ceil(log2(d)) + (wt - 1) * (ceil(log2(d)) + secpar) + wt bits, but when bd > 1, we also
    require an additional wt * (ceil(log2(bd)) + secpar) bits.

    :param secpar: Input security parameter
    :type secpar: int
    :param d: Input degree
    :type d: int
    :param bd: Input bound
    :type bd: int
    :param wt: Input weight
    :type wt: int
    :param val: Input bitstring
    :type val: str

    :return: Return a dict with integer keys and values, with wt distinct keys and all values in [-bd, ..., bd]
    :rtype: Dict[int, int]
    """
    lgd: int = ceil(log2(d))
    lgbd: int = ceil(log2(bd))
    if (bd == 1 and len(val) < lgd + (wt - 1) * (lgd + secpar) + wt) or (
            bd > 1 and len(val) < lgd + (wt - 1) * (lgd + secpar) + wt * (1 + lgbd + secpar)):
        raise ValueError('Bitstring not long enough to decode a polynomial.')
    elif not is_bitstring(val):
        raise ValueError('Cannot decode polynomial coefficient indices from a non-bitstring.')
    x_for_indices: str = val[:ceil(log2(d)) + (wt - 1) * (ceil(log2(d)) + secpar)]
    x_for_coefficients: str = val[ceil(log2(d)) + (wt - 1) * (ceil(log2(d)) + secpar):]
    indices: List[int] = decode2indices(secpar, d, wt, x_for_indices)
    coefs: List[int] = decode2coefs(secpar, bd, wt, x_for_coefficients)
    return {index: coefficient for index, coefficient in zip(indices, coefs)}


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
            Powers of rou for use in computing NTT/INTT
        zetas_invs: List[int]
            Powers of rou for use in computing NTT/INTT

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

    def __init__(self, pars: dict):
        """
        Create a new LatticeParameters object with input pars, compute the rou, rou_inv, zetas, and zetas_inv

        TODO: should we replace input parameters with Any?

        :param pars: Input parameters
        :type pars: dict
        """
        if 'degree' not in pars or 'length' not in pars or 'modulus' not in pars:
            raise ValueError('LatticeParameters requires degree, length, and modulus.')
        elif not isinstance(pars['degree'], int):
            raise ValueError('LatticeParameters requires integer degree.')
        elif not isinstance(pars['length'], int):
            raise ValueError('LatticeParameters requires integer length.')
        elif not isinstance(pars['modulus'], int):
            raise ValueError('LatticeParameters requires integer modulus.')
        elif not pars['degree'] > 0:
            raise ValueError('LatticeParameters requires strictly positive degree.')
        elif not pars['length'] > 0:
            raise ValueError('LatticeParameters requires strictly positive length.')
        elif not pars['modulus'] > 1:
            raise ValueError('LatticeParameters requires modulus > 1')
        elif sum(int(i) for i in bin(pars['degree'])[2:]) != 1:  # should this just use is_pow_two()+
            raise ValueError('LatticeParameters requires power-of-two degree.')
        elif not is_ntt_friendly_prime(pars['modulus'], pars['degree']):
            raise ValueError('LatticeParameters requires an NTT-friendly prime.')

        self.degree = pars['degree']
        self.length = pars['length']
        self.modulus = pars['modulus']
        self.halfmod = self.modulus // 2
        self.logmod = ceil(log2(self.modulus))
        self.n: int = 2 * self.degree
        self.lgn: int = ceil(log2(self.n))
        self.rou, self.rou_inv = get_prim_rou_and_rou_inv(q=self.modulus, d=self.degree)
        self.zetas, self.zetas_invs = make_zetas_and_invs(q=self.modulus, d=self.degree, halfmod=self.halfmod,
                                                          logmod=self.logmod, n=self.n, lgn=self.lgn)

    def __eq__(self, other) -> bool:
        """
        Compare two LatticeParameters objects for equality. We only check degree, length, and modulus. This is due
        to the fact that all the other parameters are derived from these three.

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


def get_gen_bits_per_poly(secpar: int, lp: LatticeParameters, wt: int, bd: int) -> int:
    """
    Compute bits required to decode a random bitstring to a polynomial of a certain weight and bound given some
    lattice parameters with a negligible bias away from uniformity. Note that this is not the same as the number of bits
    required to store a polynomial from the lattice! We require many more bits to sample, in order to ensure the
    distribution is negligibly different from uniform.

    Assumes a polynomial (which is a sum of monomials) is represented only by the coefficients and exponents of the non-
    zero summands. For example, to describe f(X) = 1 + 2 * X ** 7 + 5 * X ** 15, we can just store (0, 1), (7, 2), and
    (15, 5). If the polynomial has weight wt, then we store wt pairs. Each pair consists of an exponent from 0, 1, ...,
    d-1, and each coefficient is an integer from [-((bd-1)//2, ..., (bd-1)//2]. In fact, to sample the first monomial
    exponent from [0, 1, ..., d - 1] with zero bias requires only ceil(log2(d)) bits, but sampling each subsequent
    monomial exponent requires an additional secpar bits. And, to sample the coefficient with negligible bias from
    uniform requires 1 + ceil(log2(bd)) + secpar bits.

    :param secpar: Security parameter (necessary for bias)
    :type secpar: int
    :param lp: Lattice parameters
    :type lp: LatticeParameters
    :param wt: Polynomial weight (number of non-zero coefficients)
    :type wt: int
    :param bd: Polynomial norm bound (maximum of absolute values of coefficients).
    :type bd: int

    :return: Bits required to decode to a polynomial without bias.
    :rtype: int
    """
    outbytes_per_poly: int = ceil(log2(lp.degree))
    outbytes_per_poly += (wt - 1) * (log2(lp.degree) + secpar) + wt * (1 + ceil(log2(bd)) + secpar)
    return ceil(outbytes_per_poly / 8.0)


class Polynomial(object):
    """
    Hybrid object with some container, some callable, and some numeric properties. Instantiated with a coefficient
    representation of a polynomial using the representation described in the comments of get_n_per_poly. On
    instantiation, the NTT value representation of the coefficient representation is computed and stored in self.vals
    and the coefficients are forgotten; all arithmetic is done with the NTT values.

    Attributes
    ----------
        pars: LatticeParameters
            LatticeParameters object for use in all arithmetic.
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
        get_coefs()
            Recompute and output the coefs from ntt_values by calling ntt with inv_flag=True
        norm()
            Return the maximum of the absolute value of the values in coefs
        weight()
            Return the length of coefs.
        norm_and_weight()
            Return a tuple with norm and weight.
    """
    lp: LatticeParameters
    ntt_representation: List[int]

    def __init__(self, pars: LatticeParameters, coefs: Dict[int, int]):
        """
        Initialize a polynomial object by passing in a LatticeParameters object and coefs, which is a Dict[int, int]
        where keys are monomial exponents and values are coefficients.

        :param pars: Input LatticeParameters
        :type pars: LatticeParameters
        :param coefs: Coefficient dictionary
        :type coefs: Dict[int, int]
        """
        if len(coefs) > pars.degree:
            raise ValueError('Cannot create polynomial with too many coefficients specified.')
        elif any(i >= pars.degree for i in coefs):
            raise ValueError('Cannot create polynomial with any monomial whose power exceeds the degree.')
        elif any(i < 0 for i in coefs):
            raise ValueError('Cannot create polynomial with any monomial whose power is negative.')
        self.lp = pars
        self._reset_vals(coefs=coefs)

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
        x = self.coefficient_representation_and_norm_and_weight()
        y = other.coefficient_representation_and_norm_and_weight()
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
        result.ntt_representation = [
            cent(q=self.lp.modulus, halfmod=self.lp.halfmod, logmod=self.lp.logmod, val=x + y) for x, y in
            zip(result.ntt_representation, other.ntt_representation)]
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
        result.ntt_representation = [
            cent(q=self.lp.modulus, halfmod=self.lp.halfmod, logmod=self.lp.logmod, val=x - y) for x, y
            in zip(result.ntt_representation, other.ntt_representation)]
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
        result.ntt_representation = [
            cent(q=self.lp.modulus, halfmod=self.lp.halfmod, logmod=self.lp.logmod, val=x * y) for x, y
            in zip(result.ntt_representation, other.ntt_representation)]
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

        TODO: Refactor to use self.vals instead, to save computing the NTT for a string representation

        :return:
        :rtype: str
        """
        coef_rep, norm, wt = self.coefficient_representation_and_norm_and_weight()
        sorted_keys = sorted(list(coef_rep.keys()))
        sorted_coefs = [(i, coef_rep[i]) for i in sorted_keys]
        return str((sorted_coefs, norm, wt))

    def _ntt(self, inv_flag: bool, val: List[int]) -> List[int]:
        """
        Very thin wrapper that attaches ntt method to the Polynomial object

        :param inv_flag: Indicates whether we are performing forward NTT or inverse NTT
        :type inv_flag: bool
        :return: Return the NTT (or INTT) of the inputs x.
        :rtype: List[int]
        """
        return ntt(q=self.lp.modulus, zetas=self.lp.zetas, zetas_inv=self.lp.zetas_invs, val=val,
                   halfmod=self.lp.halfmod, logmod=self.lp.logmod, n=self.lp.n, lgn=self.lp.lgn,
                   inv_flag=inv_flag)

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

    def coefficient_representation_and_norm_and_weight(self) -> Tuple[Dict[int, int], int, int]:
        """
        Compute the coefficient representation of the polynomial by performing the inverse NTT on self.vals, compute the
        norm and the wight, and return all these..

        :return: Coefficient representation of the Polynomial, norm, and weight.
        :rtype: Tuple[Dict[int, int], int, int]
        """
        tmp: List[int] = self._ntt(inv_flag=True, val=self.ntt_representation)
        left: List[int] = tmp[:self.lp.degree]
        right: List[int] = tmp[self.lp.degree:]
        coefs: List[int] = [cent(q=self.lp.modulus, halfmod=self.lp.halfmod, logmod=self.lp.logmod, val=x - y) for
                            x, y in zip(left, right)]
        coefs_dict: Dict[int, int] = {index: value for index, value in enumerate(coefs) if value != 0}
        norm = max(abs(coefs_dict[value]) for value in coefs_dict)
        weight = len(coefs_dict)
        return coefs_dict, norm, weight


def randpoly(lp: LatticeParameters, bd: int = None, wt: int = None) -> Polynomial:
    """
    Generate a random polynomial with norm bounded by bd and weight bounded by wt. Relies on randbelow and randbits
    from the secrets library to generate data. Since the secrets library is thought to be secure for cryptographic use,
    the results should have a negligible bias away from uniformity.

    :param lp: Input lattice parameters
    :type lp: LatticeParameters
    :param bd: Input bound on coefficients
    :type bd: int
    :param wt: Input weight on the polynomial
    :type wt: int
    :return:
    :rtype: Polynomial
    """
    f_coefs: Dict[int, int] = dict()
    n: int = max(1, min(bd, lp.modulus // 2))
    w: int = max(1, min(wt, lp.degree))
    while len(f_coefs) < w:
        next_index: int = randbelow(lp.degree)
        while next_index in f_coefs:
            next_index = randbelow(lp.degree)
        next_val: int = randbelow(n) + 1
        next_val *= (2 * randbits(1) - 1)
        f_coefs[next_index] = next_val

    return Polynomial(pars=lp, coefs=f_coefs)


class PolynomialVector(object):
    """
    Contains LatticeParameters and a list of polynomials. WARNING: We repurpose the pow notation for scaling by a poly.

    Attributes
    ----------
        pars: LatticeParameters
            LatticeParameters object for use in all arithmetic.
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
            Scale self by other. Abuse of notation.
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
    entries: List[Polynomial]

    def __init__(self, pars: LatticeParameters, entries: List[Polynomial]):
        """
        Instantiate with some input LatticeParameters and a list of Polynomial entries.

        :param pars: Input lattice parameters
        :type pars: LatticeParameters
        :param entries: Input polynomial entries
        :type entries: List[Polynomial]
        """
        self.lp = pars
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
        :return: The dot product of self against other.
        :rtype: PolynomialVector
        """
        if self.lp != other.lp:
            raise ValueError('Can only compute dot products with polynomials with the same parameters.')
        each_product = [x * y for x, y in zip(self.entries, other.entries)]
        return sum(each_product)

    def __pow__(self, scalar: Polynomial):
        """
        Scale a PolynomialVector by a Polynomial scalar. So if x is a PolynomialVector and y is a Polynomial,
        z = x ** y gives us the PolynomialVector such that the ith entry is z.entries[i] == x.entries[i] * y.

        :param scalar: Input polynomial for scaling
        :type scalar: Polynomial
        :return:
        :rtype: PolynomialVector
        """
        # Abuse ** operator to scale vectors: cv = v**c
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

    def coefficient_representation_and_norm_and_weight(self) -> List[Tuple[Dict[int, int], int, int]]:
        """
        Calls coefficient_representation_and_norm_and_weight for each entry.

        :return:
        :rtype: List[Tuple[Dict[int, int], int, int]]
        """
        return [val.coefficient_representation_and_norm_and_weight() for i, val in enumerate(self.entries)]


def randpolyvec(lp: LatticeParameters, bd: int = None, wt: int = None) -> PolynomialVector:
    """
    Generate a random PolynomialVector with bounded Polynomial entries. Essentially just instantiates a
    PolynomialVector object with a list of random Polynomial objects as entries, which are in turn generated by randpoly

    :param lp: Input lattice parameters
    :type lp: LatticeParameters
    :param bd: Input bound
    :type bd: int
    :param wt: Input weight
    :type wt: int
    :return:
    :rtype: PolynomialVector
    """
    if bd is None or not isinstance(bd, int) or bd < 1:
        raise ValueError('Cannot generate a random polynomial vector without a positive integer bound')
    elif wt is None or not isinstance(wt, int) or wt < 1:
        raise ValueError('Cannot generate a random polynomial vector without a positive integer weight')
    return PolynomialVector(pars=lp, entries=[randpoly(lp=lp, bd=bd, wt=wt) for _ in range(lp.length)])


def hash2bddpoly(secpar: int, lp: LatticeParameters, bd: int, wt: int, salt: str, m: str) -> Polynomial:
    """
    Hash an input message msg and salt to a polynomial with norm bound at most bd and weight at most wt.

    :param secpar: Input security parameter
    :type secpar: int
    :param lp: Input lattice parameters
    :type lp: LatticeParameters
    :param bd: Input bound
    :type bd: int
    :param wt: Input weight
    :type wt: int
    :param m: Input string
    :type m: str
    :param salt: Input salt
    :type salt: str
    :return:
    :rtype: Polynomial
    """
    num_bits_for_hashing: int = get_gen_bits_per_poly(secpar=secpar, lp=lp, wt=wt, bd=bd)
    new_binary_digest: str = binary_digest(m, num_bits_for_hashing, salt)
    coefs: Dict[int, int] = decode2polycoefs(secpar=secpar, d=lp.degree, bd=bd, wt=wt, val=new_binary_digest)
    return Polynomial(pars=lp, coefs=coefs)


def hash2bddpolyvec(
        secpar: int, lp: LatticeParameters, bd: int, wt: int, salt: str, num_entries: int, m: str
) -> PolynomialVector:
    """
    Hash an input message msg and salt to a polynomial vector with norm bound at most bd and weight at most wt. Just
    calls decode2polycoefs repeatedly.

    :param num_entries:
    :type num_entries:
    :param secpar: Input security parameters
    :type secpar: int
    :param lp: Input lattice parameters
    :type lp: LatticeParameters
    :param bd: Input bound
    :type bd: int
    :param wt: Input weight
    :type wt: wt
    :param m: Input message
    :type m: str
    :param salt: Input salt
    :type salt: str
    :return: Call decode2polycoefs for length, create Polynomial for each, return a PolynomialVector with these entries
    :rtype: PolynomialVector
    """
    # Note: this can't exploit hash_to_bounded_polynomial, which hashes the message independently.
    num_bits: int = get_gen_bits_per_poly(secpar=secpar, lp=lp, wt=wt, bd=bd)
    total_bits: int = num_entries * num_bits
    x: str = binary_digest(m, total_bits, salt)
    xs: List[str] = [x[i * 8 * num_bits: (i + 1) * 8 * num_bits] for i in range(num_entries)]
    coefs: List[Dict[int, int]] = [decode2polycoefs(secpar, lp.degree, bd, wt, i) for i in xs]
    entries: List[Polynomial] = [Polynomial(pars=lp, coefs=i) for i in coefs]
    return PolynomialVector(pars=lp, entries=entries)
