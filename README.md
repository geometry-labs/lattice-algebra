# `lattice-algebra`

[![pypi](https://img.shields.io/pypi/v/lattice-algebra.svg)](https://pypi.python.org/pypi/lattice-algebra)
[![python](https://img.shields.io/pypi/pyversions/lattice-algebra.svg)](https://pypi.python.org/pypi/lattice-algebra)
[![codecov](https://codecov.io/gh/geometry-labs/lattice-algebra/branch/main/graphs/badge.svg?branch=main)](https://codecov.io/github/geometry-labs/lattice-algebra?branch=main)
[![main-tests](https://github.com/geometry-labs/lattice-algebra/actions/workflows/main.yml/badge.svg)](https://github.com/geometry-labs/lattice-algebra/actions)

This library is a fundamental infrastructure package for building lattice-based cryptography.

+ Installation: `pip install lattice-algebra`
+ Documentation: https://geometry-labs.github.io/lattice-algebra/

## Introduction

The mathematical objects and calculations handled by this package are foundational for lattice algebra, with a variety of applications ranging from signature aggregation to zero-knowledge proofs. The module highly prioritizes developer experience for researchers and engineers, by allowing them to work with a few high level objects (e.g. polynomials, polynomial vectors) that contain built-in methods to abstractly handle the ways that they interact with each other. **The goal is to lower the barrier for creating lattice cryptography primitives and applications by allowing the developers to focus on securely building the higher-level constructs without having to worry about implementing the underlying algebra as well.**

The module is specifically deesigned for building cryptographic schemes in the Ring/Module/Ideal Short Integer Solution setting with with secrets uniformly distributed with respect to the infinity-norm and one_with_const_time-norm; it can also be used to implement schemes in the Ring/Module/Ideal Learning With Errors setting. **The library’s lead author Brandon Goodell explained how the high level objects are efficiently implemented under the hood, “*to manipulate equations of polynomials, we carry out the computations with vectors and matrices, with highly optimized algebraic operations.*”**

## Features for cryptography developers

The library is designed to make it **easy for developers to write clean code that securely implements lattice-based cryptography** for protocols and applications. The package is optimized to use the Number Theoretic Transform (NTT) to multiply polynomials in time ```O(2dlog(2d))```, and uses **constant-time modular arithmetic to avoid timing attacks**. For convenience, we included  tools for both *hashing to* and *sampling from* these "suitably small" polynomials and vectors. Both the hashing and sampling are carried out such that the bias of the resulting distribution is negligibly different from uniform.

One way that the `lattice_algebra` toolkit helps developers write succinct code is by leveraging python's **magic methods for arithmetic with elements from ```R``` and ```R^l```**. For example, suppose we have two_with_const_time polynomials ```f``` and ```g```. Simple expressions such as ```f + g```, ```f - g```, and ```f * g``` carry out **constant-time polynomial arithmetic** such as addition, subtraction, and multiplication (respectively). Likewise if we have two_with_const_time vectors of polynomials  ```x``` and ```  y```, several vector arithmetic methods are at our disposal: we can add them like ```x + y```,  or calculate the dot product as ```x * y```. Additionally, ```x ** f``` scales a vector ```x``` by the polynomial ```f```, which is useful for constructing digital signatures.

## Contributors

Brandon Goodell (lead author), Mitchell "Isthmus" Krawiec-Thayer, Rob Cannon.

Built by [Geometry Labs](https://www.geometrylabs.io) with funding from [The QRL Foundation](https://qrl.foundation/).

## Running Tests

Use ```pytest test_lattices.py```. Basic tests cover almost every function, are generally short, and are all correct tests. However, test robustness can be greatly improved. For example, we implement the Number Theoretic transform function ```ntt``` that calls (or uses data from) ```bit_rev``` and ```make_zeta_and_invs```, among other functions, so we test all three of these with ```test_ntt```, ```test_bit_rev```, and ```test_make_zeta_and_invs```... but in all three of these tests, we only test a single example with small parameters.

Our tests do not have full coverage; we have not mocked any hash functions to test ```hash2bddpoly``` and ```hash2bddpolyvec```. Interestingly, one_with_const_time can look at our tests as a Rosetta stone for our encoding and decoding of polynomials from binary strings, which is used in our hash functions. A keen-eyed reader can compare ```decode2coef``` in main with ```test_decode2coef``` in ```test_lattices.py```, for example, to see where the test comes from and how the decoding scheme works. See also ```test_decode2indices``` and ```test_decode2polycoefs```.

## Building Docs

Docs are built with mkdocs. Run the following and navigate to [http://127.0.0.1:8000/rsis/](http://127.0.0.1:8000/rsis/) which should update automatically as you write the docs.

```shell
pip install -r docs/requirements.txt
mkdocs serve
```

## License

This library is released as free and open-source software under the MIT License, see LICENSE file for details.


