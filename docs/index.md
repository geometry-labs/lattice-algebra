# `lattice-algebra`

Infrastructure package for lattice-based crypto.

## Installation and Building Docs

To install this library, run:

```shell
pip install lattice-algebra
```

Docs are built with mkdocs; first use `pip install -r docs/requirements.txt`, 
then use `mkdocs serve`, then navigate to [http://127.0.0.1:8000/rsis/](http://127.0.0.1:8000/rsis/).

# Introduction and Background

The mathematical objects and calculations handled by this package are foundational for lattice algebra, with a variety
of applications ranging from signature aggregation to zero-knowledge proofs. The module highly prioritizes developer
experience for researchers and engineers, by allowing them to work with a few high level objects (e.g. polynomials,
polynomial vectors) that contain built-in methods to abstractly handle the ways that they interact with each other. **
The goal is to lower the barrier for creating lattice cryptography primitives and applications by allowing the
developers to focus on securely building the higher-level constructs without having to worry about implementing the
underlying algebra as well.**

The module is specifically deesigned for building cryptographic schemes in the Ring/Module/Ideal Short Integer Solution
setting with with secrets uniformly distributed with respect to the infinity-norm and one-norm; it can also be used to
implement schemes in the Ring/Module/Ideal Learning With Errors setting. High level objects are efficiently implemented
under the hood: to manipulate equations of polynomials, we carry out the computations with vectors and matrices, with
optimized algebraic operations.

## Some Math

In many lattice-based cryptographic schemes, primitives are constructed from polynomials in the
ring ```R = Zq[X]/(X^d + 1)``` where we denote the integers modulo a prime ```q``` with ```Zq```, with a degree```d```
that is a power of two such that ```(q-1) % (2*d) = 0```. Keys are often vectors from the vector space ```V=R^l``` or
matrices with entries from ```V = R^(k * l)``` for dimensions ```k```, ```l```. For example, the CRYSTALS-Dilithium
scheme sets ```q = 2^23 - 2^13 + 1```, ```d = 256```, and uses ```4x4```, ```6x5```, and ```8x7``` matrices, depending
on security level.

For certain choices of ```d```, ```q```, and ```l```, it is thought to be hard to find any vector (or matrix) ```x```
that is small enough (in terms of one or more norms on the ring ```R```) such that some matrix equation ```A * x = 0```
is satisfied, where ```A``` is a suitably random challenge from ```V```. From this hardness assumption, the map carrying
suitably small vectors (or matrices)  ```x``` to their images ```A * x``` is a one-way function. **Simulation-based
security proofs in the lattice setting are based on extracting a suitably small vector or matrix (called a *witness*)
that satisfies some system of linear equations. Overall security of the scheme is based on how small the adversary can
make this witness in terms of the norm of the witness.**

The infinity-norm and the one-norm are of particular interest: the infinity-norm of a polynomial is the absolute maximum
coefficient, and the one-norm is the absolute sum of coefficients. We can extend this definition to vectors by taking
the maximum norm of the entries of the vector. We note that if we count only the weight of a polynomial, in terms of the
number of non-zero coefficients, then we have that ```one_norm <= infinity_norm * weight```. Consequently, bounding the
infinity norm and the weight of a polynomial also has the effect of bounding the infinity norm and the one-norm. Taking
into account both the infinity norm and the weight of the polynomial (number of non-zero entries) enables tighter
inequalities that lead to smaller witnesses. This means we can achieve the **same security level with smaller
parameters** (the CRYSTALS-Dilithium scheme is an exemplary implementation of this technique).

Nothing in `lattice-algebra` limits which hardness assumptions are underlying the cryptographic scheme being
constructed. Since the library merely handles polynomials from ```R``` and vectors from ```V=R^l```, **schemes based on
other hardness assumptions (such as the Ring Learning With Errors assumption) that take place over the same ring can be
securely implemented as well**.

## Designed for cryptography developers

The library is designed to make it **easy for developers to write clean code that securely implements lattice-based
cryptography** for protocols and applications. The package is optimized to use the Number Theoretic Transform (NTT) to
multiply polynomials in time ```O(2dlog(2d))```, and uses **constant-time modular arithmetic to avoid timing attacks**.
For convenience, we included tools for both *hashing to* and *sampling from* these "suitably small" polynomials and
vectors. Both the hashing and sampling are carried out such that the bias of the resulting distribution is negligibly
different from uniform.

One way that the `lattice-algebra` toolkit helps developers write succinct code is by leveraging python's **magic
methods for arithmetic with elements from ```R``` and ```R^l```**. For example, suppose we have two polynomials ```f```
and ```g```. Simple expressions such as ```f + g```, ```f - g```, and ```f * g``` carry out **constant-time polynomial
arithmetic** such as addition, subtraction, and multiplication (respectively). Likewise if we have two vectors of
polynomials  ```x``` and ```  y```, several vector arithmetic methods are at our disposal: we can add them
like ```x + y```, or calculate the dot product as ```x * y```. Additionally, ```x ** f``` scales a vector ```x``` by the
polynomial ```f```, which is useful for constructing digital signatures.

# Intended Usage

This package handles three fundamental objects: LatticeParameters, Polynomial, and PolynomialVector. The Polynomial and
PolynomialVector objects have a LatticeParameters attribute, and the package handles computations with Polynomial and
PolynomialVector objects with matching LatticeParameters.

## LatticeParameters

The LatticeParameters class contains attributes describing the ring ```R```, namely the degree ```d```, the module
length ```l```, and the modulus ```q```. From these, additional data are pre-computed for use in various algorithms
later. We instantiate a LatticeParameters object by specifying the degree, length, and modulus in the following way.

```lp = LatticeParameters(degree=2**10, length=2**4, modulus=12289)```

We must instantiate LatticeParameters objects by passing in degree, length, and modulus. These must all be positive
integers such that the degree is a power of two and ```(modulus - 1) % (2 * degree) == 0``` otherwise a ValueError is
raised.

## Polynomial

### Polynomial Attributes and Instantiation

The Polynomial and PolynomialVector objects have a LatticeParameters attribute, ```par```, and the package handles
computations with Polynomial and PolynomialVector objects with matching LatticeParameters.

Other than the LatticeParameters object attached to each Polynomial, the Polynomial object also has
an ```ntt_representation``` attribute, which is a list of integers. To instantiate a Polynomial, we pass in the
coefficient representation of the polynomial as a dictionary of key-value pairs, where the keys are integers in the
set ```[0, 1, ..., degree - 1]``` and the value associated with a key is the coefficient of the associated monomial,
which is assumed to be a representative of an equivalence class of integers modulo ```modulus```. The coefficients are
centralized to be in the list ```[-(modulus//2), -(modulus//2)+1, ..., modulus//2 - 1, modulus//2]``` with constant-time
modular arithmetic.

For example, if ```modulus = 61``` and we want to represent ```3 * X**2 + 9 * X + 17```, we see the coefficient on the
monomial ```X**0 = 1``` is ```17```, the coefficient on the monomial ```X``` is ```9```, and the coefficient on the
monomial ```X**2``` is ```3```. So we can pack the coefficient representation of this polynomial into a dictionary
like ```{0: 17, 1: 9, 2: 3}```. So, to create a Polynomial object representing this polynomial, we use the following.

```f = Polynomial(pars=lp, coefs={0: 17, 1: 9, 2: 3})```

### Polynomial Addition, Subtraction, and Multiplication

Polynomials have support for ```__add__```, ```__radd__```, ```__sub__```, ```__mul__```, and ```__rmul__```. Thus, for
two polynomials, say ```f``` and ```g```, we simply use ```f + g```, ```f - g```, and ```f*g``` for addition,
subtraction, and multiplication. Arithmetic for these operations take place coordinate-wise with
the ```ntt_representation``` list, so they are very fast.

### Polynomial Norm, Weight, and String Representation

Polynomials have a ```cooefficient_representation_and_norm_and_weight``` method, which inverts
the ```ntt_representation``` list to obtain the coefficient representation of the polynomial, and returns this
coefficient representation together with the infinity norm and the Hamming weight of the polynomial.

The package uses ```__repr___``` to cast the output of ```coefficient_representation_and_norm_and_weight``` as a string.

#### WARNING

Computing the ```ntt_representation``` requires computing the NTT of the polynomial, and
calling ```coefficient_representation_and_norm_and_weight``` requires computing the inverse NTT of the polynomial. These
are expensive operations compared to arithmetic. Hence, _creating polynomials_, _printing them to strings_, and
_computing the norm and weight_ of polynomials should be done sparingly!

## PolynomialVector

### PolynomialVector Attributes and Instantiation

The Polynomial and PolynomialVector objects have a LatticeParameters attribute, ```par```, and the package handles
computations with Polynomial and PolynomialVector objects with matching LatticeParameters.

Other than the LatticeParameters object attached to each PolynomialVector has an ```entries``` attribute, which is just
a list of Polynomial objects. To instantiate a PolynomialVector, we pass in a list of Polynomial objects as the entries.

For example, if ```f``` is the Polynomial from the previous section and ```g(X) = -17 + 12 * X ** 2```, we can
create ```g``` and create a PolynomialVector object in the following way.

```
g = Polynomial(pars=lp, coefs={0: -17, 2: 12})
v = PolynomialVector(pars=lp, entries=[f, g])
```

Each Polynomial in ```entries``` must have the same LatticeParameters object as ```v``` and we must have
```len(entries) == lp.length```.

### PolynomialVector Addition, Subtraction, Scaling, and Dot Products

The package uses ```__add__```, ```__radd__```, and ```__sub__``` to define addition and subtraction between
PolynomialVector objects. This way, for two PolynomialVector objects, say ```v``` and ```w```, we can just
use ```v + w``` and ```v - w``` to compute the sum and difference, respectively.

The package uses ```__mul__``` and ```__rmul__``` to define the **dot product** between two PolynomialVector objects.
The dot product outputs a Polynomial object. For example, if ```v.entries == [f, g]``` and ```w.entries == [a, b]```,
then ```v * w``` returns ```f * a + g * b```.

The package repurposes ```__pow__``` to scale a PolynomialVector by a Polynomial. For example, if ```v.entries = [f, g]```
and ```a``` is some Polynomial object, then ```v ** a = [a * f, a * g]```. This is **not** exponentiation, although we
use the notation for exponentiation.

Hence, to compute a linear combination of PolynomialVectors whose coefficients are Polynomials, we compute the sum of 
"exponents" with something like this: ```sum(f ** a for f, a in zip(some_polynomial_vectors, some_polynomials))```. As
before, arithmetic operations are done using the ```ntt_representation``` of the involved polynomials, and are thus
quite fast.

### PolynomialVector Norm, Weight, and String Representation.

The string representation of a PolynomialVector, defined in ```__repr__``` is merely ```str(entries)```.

#### WARNING

To compute the string representation, the norm, or the weight of a PolynomialVector requires computing the same for each
entry in ```entries```. So our previous warning about the cost of computing the NTT and the inverse NTT applies here,
but with the added curse of dimensionality.

## Other Functionality

The library also contains functions ```randpoly```, ```hash2bddpoly```, ```randpolyvec```, and ```hash2bddpolyvec``` for
generating random Polynomial and PolynomialVector objects, either with system randomness or by hashing a message. The
output of these functions are uniformly random (at least up to a negligible difference) among the Polynomial and
PolynomialVector objects with a specified infinity norm bound and Hamming weight.

In order for the hash functions to work requires decoding bitstrings of certain lengths to Polynomial and
PolynomialVector objects in a way that keeps the output uniformly random (or at least with a negligible difference from
uniform). These are the functions ```decode2coeef```, ```decode2coefs```, ```decode2indices```,
and ```decode2polycoefs```.
