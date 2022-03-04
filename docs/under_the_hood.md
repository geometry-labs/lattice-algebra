
# Under the Hood

The library is designed to make it **easy for developers to write clean code that securely implements lattice-based cryptography** for protocols and applications. The package is optimized to use the Number Theoretic Transform (NTT) to multiply polynomials in time ```O(2dlog(2d))```, and uses **constant-time modular arithmetic to avoid timing attacks**. For convenience, we included tools for both *hashing to* and *sampling from* these "suitably small" polynomials and vectors. Both the hashing and sampling are carried out such that the bias of the resulting distribution is negligibly different from uniform.

One way that the `lattice-algebra` toolkit helps developers write succinct code is by leveraging python's **magic methods for arithmetic with elements from ```R``` and ```R^l```**. For example, suppose we have two polynomials ```f``` and ```g```. Simple expressions such as ```f + g```, ```f - g```, and ```f * g``` carry out **constant-time polynomial arithmetic** such as addition, subtraction, and multiplication (respectively). Likewise if we have two vectors of polynomials  ```x``` and ```  y```, several vector arithmetic methods are at our disposal: we can add them like ```x + y```, or calculate the dot product as ```x * y```. Additionally, ```x ** f``` scales a vector ```x``` by the polynomial ```f```, which is useful for constructing digital signatures.

## Class Details

This package handles three fundamental objects: LatticeParameters, Polynomial, and PolynomialVector. The Polynomial and PolynomialVector objects have a LatticeParameters attribute, and the package handles computations with Polynomial and PolynomialVector objects with matching LatticeParameters.

### LatticeParameters

The LatticeParameters class contains attributes describing the ring ```R```, namely the degree ```d```, the module length ```l```, and the modulus ```q```. From these, additional data are pre-computed for use in various algorithms later. We instantiate a LatticeParameters object by specifying the degree, length, and modulus in the following way.

```lp = LatticeParameters(degree=2**10, length=2**4, modulus=12289)```

We must instantiate LatticeParameters objects by passing in degree, length, and modulus. These must all be positive integers such that the degree is a power of two and ```(modulus - 1) % (2 * degree) == 0``` otherwise a ValueError is raised.

### Polynomial

#### Attributes and Instantiation

The Polynomial and PolynomialVector objects have a LatticeParameters attribute, ```lp```, and the package handles computations with Polynomial and PolynomialVector objects with matching LatticeParameters.

Other than the LatticeParameters object attached to each Polynomial, the Polynomial object also has an ```ntt_representation``` attribute, which is a list of integers. To instantiate a Polynomial, we pass in the coefficient representation of the polynomial as a dictionary of key-value pairs, where the keys are integers in the set ```[0, 1, ..., degree - 1]``` and the value associated with a key is the coefficient of the associated monomial, which is assumed to be a representative of an equivalence class of integers modulo ```modulus```. The coefficients are centralized to be in the list ```[-(modulus//2), -(modulus//2)+1, ..., modulus//2 - 1, modulus//2]``` with constant-time modular arithmetic.

For example, if ```modulus = 61``` and we want to represent ```3 * X**2 + 9 * X + 17```, we see the coefficient on the monomial ```X**0 = 1``` is ```17```, the coefficient on the monomial ```X``` is ```9```, and the coefficient on the monomial ```X**2``` is ```3```. So we can pack the coefficient representation of this polynomial into a dictionary like ```{0: 17, 1: 9, 2: 3}```. So, to create a Polynomial object representing this polynomial, we use the following.

```f = Polynomial(pars=lp, coefs={0: 17, 1: 9, 2: 3})```

#### Arithmetic

Polynomials support ```__add__```, ```__radd__```, ```__sub__```, ```__mul__```, and ```__rmul__```. Thus, for two polynomials, say ```f``` and ```g```, we simply use ```f + g```, ```f - g```, and ```f*g``` for addition, subtraction, and multiplication. Arithmetic for these operations take place coordinate-wise with the ```ntt_representation``` list, so they are very fast.

#### Polynomial Norm, Weight, and String Representation

Polynomials have a ```cooefficient_representation_and_norm_and_weight``` method, which inverts the ```ntt_representation``` list to obtain the coefficient representation of the polynomial, and returns this coefficient representation together with the infinity norm and the Hamming weight of the polynomial.

The package uses ```__repr___``` to cast the output of ```get_coef_rep``` as a string.

_NOTE_: Computing the ```ntt_representation``` requires computing the NTT of the polynomial, and calling ```get_coef_rep``` requires computing the inverse NTT of the polynomial. These are relatively expensive operations compared to arithmetic. Hence, _creating polynomials_, _printing them to strings_, and _computing the norm and weight_ of polynomials should be done once, after all other computations are complete.

### PolynomialVector

#### PolynomialVector Attributes and Instantiation

The Polynomial and PolynomialVector objects have a LatticeParameters attribute, ```par```, and the package handles computations with Polynomial and PolynomialVector objects with matching LatticeParameters.

Other than the LatticeParameters object attached to each PolynomialVector has an ```entries``` attribute, which is just a list of Polynomial objects. To instantiate a PolynomialVector, we pass in a list of Polynomial objects as the entries.

For example, if ```f``` is the Polynomial from the previous section and ```g(X) = -17 + 12 * X ** 2```, we can create ```g``` and create a PolynomialVector object in the following way.

```
g = Polynomial(pars=lp, coefs={0: -17, 2: 12})
v = PolynomialVector(pars=lp, entries=[f, g])
```

Each Polynomial in ```entries``` must have the same LatticeParameters object as ```v``` and we must have ```len(entries) == lp.length```.

#### PolynomialVector Addition, Subtraction, Scaling, and Dot Products

PolynomialVector objects support ```__add__```, ```__radd__```, and ```__sub__``` to define addition and subtraction between PolynomialVector objects. This way, for two PolynomialVector objects, say ```v``` and ```w```, we can just use ```v + w``` and ```v - w``` to compute the sum and difference, respectively.

The package uses ```__mul__``` and ```__rmul__``` to define the **dot product** between two PolynomialVector objects. The dot product outputs a Polynomial object. For example, if ```v.entries == [f, g]``` and ```w.entries == [a, b]```, then ```v * w``` returns ```f * a + g * b```.

The package repurposes ```__pow__``` to scale a PolynomialVector by a Polynomial. For example, if ```v.entries = [f, g]``` and ```a``` is some Polynomial object, then ```v ** a = [a * f, a * g]```. This is **not** exponentiation, although we use the notation for exponentiation.

Hence, to compute a linear combination of PolynomialVectors whose coefficients are Polynomials, we compute the sum of "exponents" with something like this: ```sum(f ** a for f, a in zip(some_polynomial_vectors, some_polynomials))```. As before, arithmetic operations are done using the ```ntt_representation``` of the involved polynomials, and are thus quite fast.

#### PolynomialVector Norm, Weight, and String Representation.

The string representation of a PolynomialVector, defined in ```__repr__``` is merely ```str(entries)```.

_NOTE_: Like for ```Polynomial```, instantiation requires computing the NTT of polynomials. So our previous warning about the cost of computing the NTT and the inverse NTT applies here, but with the added curse of dimensionality.
