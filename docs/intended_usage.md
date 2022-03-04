
# Intended Usage (and Some Background)

## Arithmetic

For some TLDR code snippets, proceed to the Example below.

In many lattice-based cryptographic schemes, primitives are constructed from polynomials in the ring ```R = Zq[X]/(X^d + 1)``` where we denote the integers modulo a prime ```q``` with ```Zq```, with a degree```d``` that is a power of two such that ```(q-1) % (2*d) = 0```. Keys are often vectors from the vector space ```V=R^l``` or matrices with entries from ```V = R^(k * l)``` for dimensions ```k```, ```l```. For example, the CRYSTALS-Dilithium scheme sets ```q = 2^23 - 2^13 + 1```, ```d = 256```, and uses ```4x4```, ```6x5```, and ```8x7``` matrices, depending on security level.

We encode the vector space ```V``` using the ```LatticeParameters``` object which holds the ```degree```, ```modulus```, and ```length``` attributes. 

We create polynomials in ```R``` by using the ```Polynomial``` object, which has a ```LatticeParameters``` attribute called ```lp```. When we instantiate a Polynomial, we pass in the lattice parameters and a *coefficient representation* of that polynomial as a dictionary. The keys determine the power on the monomial and the value determines the coefficient. 

For example, if ```degree = 8``` and ```modulus = 257```, then the polynomial ```f(X) = 1 + 256 * X + 12 * X**2``` can be created with either ```{0: 1, 1: 256, 2: 12}``` or ```{0: 1, 1: -1, 2: 12}``` as the input coefficient representation, since ```(256 - (-1)) % 257 == 0```. 

We create polynomial vectors in ```V``` by using the ```PolynomialVector``` object, which has a ```LatticeParameters``` attribute and another attribute called ```entries``` which is just a list of ```Polynomial``` objects.

For example, if ```f``` and ```g``` are ```Polynomial``` and have the same lattice parameters, which has ```l = 2```, then we can instantiate the vector ```v = [f, g]``` by passing in ```entries=[f, g]```.

From a ```Polynomial```, we can access the NTT representation at any time for almost no cost with the ```ntt_representation``` attribute. However, regaining the coefficient representation requires computing the inverse NTT, which is costly. Checking norms and weights of a ```Polynomial``` or a ```PolynomialVector``` requires the coefficient representation. We re-emphasize that this representation is costly to compute. Thus, it should only be computed once (and norms and weights should only be checked at the end of algorithms). We regain the coefficient representation of a ```Polynomial``` or a ```PolynomialVector``` object by calling the ```get_coef_rep``` function.

### Example of Arithmetic

Consider the vector space ```V``` defined with ```degree = 8```, ```modulus = 257```, and ```length = 3```, and the following 8 polynomials (which are proportional to the first seven Legendre polynomials, for a convenient example).

 1. ```a(X) = 1```
 2. ```b(X) = X```
 3. ```c(X) = -1 + 3 * X**2```
 4. ```d(X) = -3 * X + 5 * X ** 3```
 5. ```e(X) = 3 - 30 * X ** 2 + 35 * X ** 4```
 6. ```f(X) = 15 * X - 70 * X ** 3  + 63 * X ** 5```
 7. ```g(X) = -5 + 105 * X ** 2 - 315 * X ** 4 + 231 * X ** 6```
 8. ```h(X) = -35 * X + 315 * X ** 3 - 693 * X ** 5 + 429 * X ** 7```

In the following code, we instantiate the vector space ```V = R^3```, we instantiate these polynomials, we compute a few of their sums and products, we create two vectors of polynomials, ```v = [a, b, c]``` and ```u = [d, e, f]```, we compute the dot product ```v * u``` of these two vectors, we compare it to the sums and products we just computed by calling the ```get_coef_rep``` function, we scale ```v``` by ```g(X)``` and we scale ```u``` by ```h(X)```, we compute this linear combination of ```v``` and ```u```, and we print the coefficient representation, norm, and weight.

```
from lattice_algebra import LatticeParameters, Polynomial, PolynomialVector

lp = LatticeParameters(pars={'degree': 8, 'modulus': 257, 'length': 3})  # make V

a = Polynomial(lp = lp, coefs = {0: 1})  # Make 8 polynomials proportional to the first 8 Legendre polys
b = Polynomial(lp = lp, coefs = {1: 1})
c = Polynomial(lp = lp, coefs = {0: -1, 2: 3})
d = Polynomial(lp = lp, coefs = {1: -3, 3: 5})
e = Polynomial(lp = lp, coefs = {0: 3, 2: -30, 4: 35})
f = Polynomial(lp = lp, coefs = {1: 15, 3: -70, 5: 63})
g = Polynomial(lp = lp, coefs = {0: -5, 2: 105, 4: -315, 6: 231})
h = Polynomial(lp = lp, coefs = {1: -35, 3: 315, 5: -693, 7: 429})

prods = [a * d, b * e, c * f]  # We can add, subtract, multiply, and use python built-in sum()
sum_of_these = sum(prods)
coef_rep_of_sum, n_sum, n_sum = sum_of_these.get_coef_rep()

v = PolynomialVector(lp = lp, entries = [a, b, c]) # Make some polynomial vectors
u = PolynomialVector(lp = lp, entries = [d, e, f])

dot_product = v * u  # We can compute the dot product, which should match the sum above

coef_rep_of_dot_prod, n_dot, w_dot = dot_product.get_coef_rep()
assert n_sum == n_dot
assert w_sum == w_dot
assert list(coef_rep_of_dot_prod.keys()) == list(coef_rep_of_sum.keys())
for next_monomial in coef_rep_of_dot_prod:
    assert (coef_rep_of_dot_prod[next_monomial] - coef_rep_of_sum[next_monomial]) % lp.modulus == 0

scaled_v = v ** g  # We can also scale a vector by a polynomial with __pow__
scaled_u = u ** h
lin_combo = scaled_v + scaled_u  # We can add vectors (and subtract!)
also_lin_combo = sum([i ** j for i, j in zip([v, u], [g, h])])  # more pythonically
assert also_lin_combo == lin_combo

# Lastly, let's print the coefficient representation, norm, and weight of this lin combo
coef_rep, n, w = lin_combo.get_coef_rep()
print(f"Coefficient representation of linear combination = {coef_rep}")
print(f"Norm of linear combination = {n}")
print(f"Weight of linear combination = {w}")
```

## Randomness and Hashing

The library also contains functions ```random_polynomial```, ```hash2bddpoly```, ```random_polynomialvector```, and ```hash2bddpolyvec``` for generating random ```Polynomial``` and ```PolynomialVector``` objects, either with system randomness or by hashing a message. The output of these functions are uniformly random (at least up to a negligible difference) among the ```Polynomial``` and ```PolynomialVector``` objects with a specified infinity norm bound and Hamming weight. Randomness is generated using the ```secrets``` module.

### Example of Randomness and Hashing.

In the following code, we first use the salt ```'SOME_SALT'``` to hash the string ```hello world``` to an instance of the ```Polynomial``` class, say ```x```, and an instance of the ```PolynomialVector``` class, say ```v```. In both cases, the polynomials in the hash output should have at most ```4``` non-zero coefficients since ```wt = 4```, and all of those should be in the list ```[-1, 0, 1]``` since ```bd = 1```. Then, we sample a new random ```Polynomial```, say ```y```, and a new random ```PolynomialVector```, say ```u```, using ```random_polynomial``` and ```random_polynomialvector```, respectively. Note that there are around ```2 ** 12``` possible outputs of ```random_polynomial``` and ```hash2bddpoly``` using these parameters, and around ```2 ** 36``` possible outputs of ```random_polynomialvector``` and ```hash2bddpolyvec```. In particular, the chance that we obtain ```x == y``` and ```v == u``` under these conditions is around ```2 ** -48```. While this is not cryptographically small, it is pretty durned small, so the following code should pass assertions.  

```
from lattice_algebra import hash2bddpoly, hash2bddpolyvec, random_polynomial, random_polynomialvector

lp = LatticeParameters(pars={'degree': 8, 'modulus': 257, 'length': 3})  # make V

x = hash2bddpoly(secpar = lp.secpar, lp = lp, bd = 1, wt = 4, salt = 'SOME_SALT', m='hello world')
coef_rep, n, w = x.get_coef_rep()
assert n <= 1  # should always pass
assert len(coef_rep) <= w <= 4  # should always pass

v = hash2bddpoly(secpar = lp.secpar, lp = lp, bd = 1, wt = 4, salt = 'SOME_SALT', m='hello world')
coef_rep, n, w = v.get_coef_rep()
assert n <= 1  # should always pass
assert len(coef_rep) <= w <= 4  # should always pass

y = random_polynomial(secpar = lp.secpar, lp = lp, bd = 1, wt = 4)
coef_rep, n, w = y.get_coef_rep()
assert n <= 1  # should always pass
assert len(coef_rep) <= w <= 4  # should always pass

u = random_polynomialvector(secpar = lp.secpar, lp = lp, bd = 1, wt = 4)
coef_rep, n, w = u.get_coef_rep()
assert n <= 1  # should always pass
sassert len(coef_rep) <= w <= 4  # should always pass

assert x != y or v != u  # should pass with probability 1 - 2 ** - 48
```

In order for the hash functions to work requires decoding bitstrings of certain lengths to ```Polynomial``` and ```PolynomialVector``` objects in a way that keeps the output uniformly random (or at least with a negligible difference from uniform). These are the functions ```decode2coeef```, ```decode2coefs```, ```decode2indices```, and ```decode2polycoefs```.

