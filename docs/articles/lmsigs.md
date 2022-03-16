# Zen and the Art of Compact Post-Quantum Finance

Cryptographic schemes that are secure against quantum adversaries are infamous for their relative inefficiency in
comparison to their classically secure counterparts. Nevertheless, the days are numbered for classical cryptography as
quantum computers loom on the horizon. This naturally leads our interest toward cryptographic schemes and protocols that
are designed for quantum resistance. Especially with the advent of cryptocurrencies, the race to develop robust and
commercially viable quantum computers has a tantalizing and profitable prize awaiting the winners, and a highly
disruptive economic future for the rest of us. Hence, it is of critical importance to develop quantum-resistant
cryptography tools and to begin migration to quantum-resistant settings as soon as possible, in the spirit of Quantum
Resistant Ledger. This can mean sacrificing the efficiency of classical cryptography, but (of course!) a robust and
secure cryptographic system is useless if it is too unwieldy to use.

This is a multipart article series exploring avenues toward more efficient quantum-resistant transacting. Here in part
one, we explore a candidate lattice-based one-time signature scheme. The scheme is rather similar in certain ways to the
NIST post-quantum standard candidate, CRYSTALS-Dilithium, and provides an avenue to discuss optimizations available
toward smaller keys and signatures in lattice-based cryptographic schemes. In the next part, we cover the technical end
of _signature aggregation_, at least from one angle. The third part of this series explores _payment channels_
constructed with _adaptor signatures_, their security models, and their implementations. In the final part, we describe
a blockchain-agnostic code-based approach to trustlessly (or, rather, _trust-minimally_) reducing local storage
requirements for network participants while keeping bootstrapping costs low.

## Part One: In Pursuit of More Efficient Lattice-Based Signing

In the following article, we remind the reader that if an algorithm $A$ takes input $x$ and outputs $y$, we say that the
output of an implementation of $A$ was _honestly computed_ if $y$ was computed according to the specification of $A$,
with no additional computations performed in between steps. We say the output of the implementation was _semi-honestly
computed_ if $y$ was computed almost according to the specification of $A$, except possibly with additional computations
inserted between steps.

### Digital Signatures

One-time signatures are useful for a variety of cryptographic applications, but to talk about one-time signatures, we
first speak about usual digital signature schemes first.

A digital signature scheme is a tuple of algorithms $(\texttt{Setup}, \texttt{Keygen}, \texttt{Sign}, \texttt{Verify})$
that informally work as follows.

0. $\texttt{Setup}(\lambda) \to \rho$. Input a security parameter $\lambda$, typically indicating the number of bits of
   security, and outputs some public parameters, $\rho$. Typically, $\rho$ contains a description of a secret signing
   key set, $K_S$, a public verification key set, $K_V$, a message set $M$, and a signature set $S$. $\texttt{Setup}$ is
   usually run by all parties before participating. The input and output of $\texttt{Setup}$ are both public.
1. $\texttt{Keygen}(\lambda, \rho) \to (sk, vk)$. Input $(\lambda, \rho)$ and output a new random keypair $(sk, vk) \in
   K_S \times K_V$, where $sk$ is a secret signing key and $vk$ is the corresponding public verification key.
2. $\texttt{Sign}(\lambda, \rho, (sk, vk), \mu) \to \xi$. Input $(\lambda, \rho, (sk, vk), \mu)$ where $\mu \in M$ is a
   message, and outputs a signature, $\xi \in S$.
3. $\texttt{Verify}(\lambda, \rho, vk, \mu, \xi) \to b \in {0, 1}$. Output a bit $b$ indicating whether the signature is
   a valid signature on the message $m$ with the public verification key $vk$.

Since $\texttt{Setup}$ is run by all parties before participating and the inputs and outputs of $\texttt{Setup}$ are
both public, and since all algorithms take $\lambda, \rho$ as input, it is admissible to neglect including $\lambda$ and
$\rho$ in the rest of our notation, and to ignore $\texttt{Setup}$ (unless the details become relevant). Also, we
generally assume that the message set $M$ is the set of all finite-length bit strings, $M = {0,1}^*$. We symbolically
represent the above digital signature scheme as follows.

1. $\texttt{Keygen} -> (sk, vk) \in K_S \times K_V$
2. For any $(sk, vk) \in K_S \times K_V$, for any $\mu \in M$, $\texttt{Sign}((sk, vk), \mu) -> \xi \in S$.
3. For any $vk \in K_V$, for any $\mu \in M$, for any $\xi \in S$, $\texttt{Verify}(vk, \mu, \xi) -> b \in {0, 1}$.

Such a scheme, as described, is rather useless, without further conditions. To see why, consider the absurd reduction of
the case that $S = {0}$. In this case, only one signature is possible, namely the zero-bit. Or, consider the case that
$\texttt{Verify}$ always returns the bit $b=0$. In this case, every "signature" comes back invalid, even if it was
honestly signed. Or, consider the case that $\texttt{Verify} always returns the $1$ bit. In this case, every signature
is valid, so forgery is easy as pie.

This leads us to the notion of the security properties of digital signatures.

### Security Properties: Correctness

Loosely, we say a signature scheme is _correct_ whenever, for any message $\mu \in M$, for any pair $(sk, vk)$ that is
output from $\texttt{Keygen}$, $\texttt{Verify}(vk, \mu, Sign((sk, vk), \mu)) = 1$. Correctness here is in analogy to
soundness in zero-knowledge protocols. When considering security properties, it is often helpful to ask "what does it
mean for a scheme to satisfy the _negation_ of this property?"

For example, the negation of the correctness definition is as follows: there exists a message $\mu$ and a keypair $(sk,
vk)$ that is output from $Keygen$ such that $Verify(vk, \mu, Sign((sk, vk), \mu)) = 0$. In this case, a scheme that is
not correct is not guaranteed to even produce signatures that pass verification!

### Security Properties: Unforgeability

Now consider what it means for a signature scheme to be _secure_; more precisely, consider the notion of unforgeability.
If Alice attempts to forge a usual digital signature without knowing Bob's key, we can formalize this situation in a
three-step process. First, Alice and Bob agree upon input and output of $\texttt{Setup}$, to recreate the common
situation that Alice and Bob have agreed upon some publicly audited and cryptanalyzed signature scheme (cf. the NIST
post-quantum signature vetting process).

Next, Bob runs $\texttt{Keygen}$ to get some _challenge keys_ $(sk, vk)$. Bob sends $vk$ to Alice.

Lastly, Alice attempts to output some message-signature pair $(\mu, \xi)$. Alice's forgery is successful if $Verify(vk,
\mu, \xi) = 1$.

The idea behind unforgeability is that any algorithm "Alice" should fail at this game except with negligible
probability.

There are two main things to note about this game. First, of course, the game is easy if Bob gives away his $sk$ to
Alice, so it is critical that Alice learns no information about $sk$ in the course of the game in order for the result
to be considered a forgery. Second, the game fails to capture the real-life situation where Alice may have seen
signatures published under Bob's key before, perhaps posted on some public bulletin board. Thus, Alice is often granted
the benefit of the doubt and is given access to a _signature oracle_. This allows her to obtain a signature from Bob's
key on any message she likes... this models the real-life situation where Alice may be able to coerce or trick Bob into
signing a message that Alice has selected. However, in this case, Alice certainly should not be allowed to win the game
by passing off an oracle-generated signature as a forgery. Moreover, Alice generally only cares about forging
signatures _for which she does not already have a signature._ Hence, Alice is generally not interested in winning the
game even by re-using any of the messages that have already been signed by the oracle.

This leads us to the following description of the _existential unforgeability experiment_. First, Alice and Bob agree
upon the input and output of $\texttt{Setup}$. Next, Bob generates $(sk, vk) \leftarrow Keygen$ and sends $vk$ to Alice.
Next, Bob grants Alice signature oracle access where she can query the oracle with any message $\mu$ and obtain a
signature $\xi$ such that $Verify(vk, \mu, \xi) = 1$. Eventually, Alice outputs a message-signature pair $(\mu, \xi)$,
succeeding if and only if the signing oracle was not queried with $mu$ and $Verify(vk, \mu, \xi) = 1$.

We then define a scheme to be _existentially unforgeable_ if, for any algorithm "Alice", the probability that Alice
succeeds at this experiment is negligible.

As before, let us consider what it means to negate the definition of existential unforgeability. Then there exists an
algorithm "Alice" that has a non-negligible probability of winning this game. This means that Alice, with no information
about Bob's $sk$ but perhaps with the ability to persuade Bob to publish some signatures on messages of Alice's choice,
can nevertheless produce a new forgery on a message Bob has never seen before.

Of course, we can expand this game to include more challenge keys; Bob could say "here are several of my verification
keys, I challenge you to construct a valid signature on any one of them." However, the number of keys provided to Alice
is always polynomially bounded in $\lambda$, and thus Alice can at best polynomially improve her probability of success.
In particular, if we can assume that Alice succeeds with only negligible probability in the one-key case, then Alice
still can only succeed with negligible probability in the case that she is given a polynomial number of challenge keys,
so we don't need to consider this generalization.

### Variations: One-Time Signature Schemes

A one-time signature scheme is just a digital signature scheme, even with the same definition of correctness, but with a
different definition of unforgeability. In one-time schemes, producing more than one signature from the same key can
reveal that key to other users, so it is only ever safe to publish one signature from any key; this makes one-time
signatures particularly useful in constructing transaction protocols, as we shall see. However, this also means that our
definition of existential unforgeability is no longer suitable. After all, in that definition, Alice can request many
signatures from the signing oracle. To rectify this, the definition of the _one-time existential unforgeability
experiment_ is nearly identical to the existential unforgeability experiment, except Alice is only granted _one_ oracle
query.

### Simplicity First: “Clever” Is Not a Compliment in Cryptography

One-time signature schemes are oftentimes very simple to describe and implement, which leads to straightforward proofs
of security properties and mistake-resistant code. In the following examples, we show a case from classically secure
cryptography and a similar case from quantum-resistant cryptography. As we can see from the descriptions below,
describing the schemes is quite simple. Thus, implementing these schemes has fewer pitfalls than implementing more "
clever" signature schemes.

#### Example: A Schnorr-Like One-Time Signature Scheme

Let $\mathbb{G}$ be an elliptic curve group, written additively, with prime order $p$, and let $G$ be generator sampled
uniformly at random. Let $F:{0,1}^* \to \mathbb{Z}/p\mathbb{Z}$ be a random oracle. The following defines a one-time
signature scheme that satisfies one-time existential unforgeability.

0. $\texttt{Setup}(\lambda) -> \rho$. Set $\rho = (\mathbb{G}, p, G, F)$.
1. $\texttt{Keygen} -> (sk, vk)$. Sample $x, y \in \mathbb{Z}/p\mathbb{Z}$ independently and uniformly at random, set
   $X := xG$, $Y := yG$, $sk := (x, y)$, and $vk := (X, Y)$. Output $(sk, vk)$.
2. $\texttt{Sign}((sk, vk), \mu) -> \xi$. Compute $c = F(vk, \mu)$, parse $(x, y) \leftarrow sk$, set $\xi := x \cdot c
    + y (mod p)$, and output $\xi$.
3. $Verify(vk, \mu, \xi) -> b$. Check that $\xi$ can be interpreted as an element of $\mathbb{Z}/p\mathbb{Z}$, compute
   $c = F(vk, \mu)$, and output $1$ if and only if $\xi G == c X + Y$.

The scheme is correct. Indeed, if $x, y, c \in \mathbb{Z}/p\mathbb{Z}$ (which is a field, and thus closed under addition
and multiplication), then so is $x \cdot c + y$. So a semi-honestly computed $\xi$ is an integer modulo $p$ since $x$,
$c$, and $y$ are integers modulo $p$. Also, since $Sign((sk, vk), m) = x \cdot c + y$, we have that $(x \cdot c + y)G =
c \cdot X + Y$. The scheme can also be shown to be unforgeable, by showing that if Alice can produce a forgery, then
Alice can also break the discrete logarithm assumption. Indeed, if Alice can produce $\xi$ that satisfies the game, and
receives an oracle-generated signature $\xi_0$, then Alice can compute $\xi - \xi_0$ easily and solve backward for $x$
and $y$. Hence, if Alice sees a discrete logarithm challenge $X$, she can randomly sample her own $y \in
\mathbb{Z}/p\mathbb{Z}$, compute $Y = yG$, and then play the unforgeability game with the challenge keys $(X, Y)$. At
the end, she will have successfully gained enough information to easily compute the discrete logarithm of $X$. Since
computing discrete logarithms is hard with classical computers, we conclude that classical forgers do not exist (or
succeed with very low probability, or take a very long time to finish their tasks).

#### Example: Extend this Schnorr-Like Approach to the Lattice Setting

We can swap out the fundamental hardness assumption above with a different hardness assumption to get a different
scheme; using a related approach, Lyubashevsky and Micciancio first proposed signatures designed (mostly) this way
in [[1]](https://eprint.iacr.org/2013/746.pdf). In a lattice setting, we ditch the elliptic curve group in favor of
lattices induced by rings of integers and their ideals. Let $\mathbb{Z}_q$ be short for the integers modulo $q$ for a
prime modulus $q$, let $d$ be a power of two integer, let $l$ be a natural number, let $R = \mathbb{Z}[X]/(X^d + 1)$ be
a quotient ring of integer-coefficient polynomials, let $R_q$ be the quotient ring $R/qR$, let $V = R^\ell$ be the
freely-generated $R$-module with length $l$, and let $V_q = R_q^\ell$ be the freely-generated $R_q$-module with length
$\ell$. Note that $V$ and $V_q$ are also vector spaces over $\mathbb{Z}_q$ of dimension $d\cdot \ell$. Elements $f(X)
\in R$ have an infinity norm; if $f(X) = f_0 + f_1 X + ... + f_{d-1} X^{d-1}$, then $\|f\|_\infty = \max(|f_i|)$. This
definition extends to $V$, which is convenient.

We can similarly extend this definition to $R_q$ and $V_q$, but at a cost. The cost we pay is in absolute homogeneity.
In a vector space $U$ over a field $F$, for any scalar $c \in F$ and any vector $u \in U$, _absolute homogeneity_ is the
property that $\|c \cdot u\| = |c| \|u\|$, where $|c|$ is computed by lifting $c$ from $F$ to $\mathbb{C}$, the complex
numbers. If we carry our definition of $\|\cdot\|_\infty$ from $R$ to $R_q$, or from $V$ to $V_q$, we cannot use this
statement to conclude anything about $c \cdot u$ when $c \in $R$ or $R_q$ except in the special cases that $c \in
\mathbb{Z}/q\mathbb{Z}$. In fact, absolute homogeneity is relaxed, and we end up with a norm-like function with _
absolute subhomogeneity_: $\| c \dot u\| \leq d \cdot |c| \|u\|$.

For lack of a better term, we shall call such functions _subhomogeneous norms_, and if $T$ is a space that admits a
sub-homogeneous norm, we shall call $T$ a _subhomogeneously normed space_.

Let $\beta_{sk}$, $\beta_{ch}$, $\beta_v \in \mathbb{N}$. For any subset $T$ of any subhomogeneously normed space, let
$B(T, \beta) = {t \in T \mid \|t\| \leq \beta}$. Again let $F$ be a random oracle but this time, let $F:{0,1}^* \to B(
R_q, \beta_{ch})$.

0. $\texttt{Setup}(\lambda) -> \rho$. Compute $d, q, l, k, \beta_{sk}, \beta_{ch}, \beta_v$ from $\lambda$, sample
   $\underline{a}$ from $V_q$ uniformly at random, and output $\rho = (d, q, l, F, k, \beta_{sk}, \beta_{ch}, \beta_v,
   \underline{a})$. The signing key set is $K_S = B(V_q, \beta_{sk}) \times B(V_q, \beta_{sk})$, the verification key
   set is $K_V = R_q \times R_q$, the message set is length $k$ bit strings $M = {0, 1}^k$, and the signature set is $S =
   B(V_q, \beta_V)$.
1. $\texttt{Keygen} -> (sk, vk)$. Sample $\underline{x}, \underline{y}$ uniformly at random and independently from $B(
   V_q, \beta_{sk})$. Define $X = \langle \underline{a}, \underline{x} \rangle$ and $Y = \langle \underline{a},
   \underline{y}\rangle$, where $\langle \cdot, cdot \rangle$ denotes the dot product between two vectors.
2. $Sign((sk, vk), \mu) -> \xi$. Compute $c = F(vk, \mu)$ and output $\xi = \underline{x} \cdot c + \underline{y}$ where
   $\cdot$ here denotes scaling the polynomial vector $\underline{x} \in B(Vq, \beta_{sk})$ with the polynomial $c \in
   B(R_q, \beta_{ch})$.
3. $Verify(vk, \mu, \xi) -> b$. Check that \xi is an element of $B(Vq, \beta_V)$, compute $c = F(vk, \mu)$, and output
   $1$ if and only if $\langle \underline{a}, \xi\rangle = X \cdot c + Y$.

This scheme may not always be correct, depending on how norms grow in $R_q$, although correctness is easily attained. In
particular, we note that since $\xi = \underline{x} \cdot c + \underline{y}$, it is both necessary and sufficient for
$\beta_v \geq \|\underline{x} \cdot c + \underline{y}\|_\infty \leq \beta_{sk}(d\beta_{ch}+1)$ for the scheme to be
correct.

Similarly to the previous scheme, this scheme can be proven unforgeable. Indeed, just like in the previous example, if
Alice can produce a forgery, she can use this forgery together with the oracle-generated signature she receives in order
to extract some keys $\underline{x}^\prime$ and $\underline{y}^\prime$. Note the apostrophes! In general, we can't
necessarily extract the keys exactly, but we can extract _some keys_ that have matching public keys!

But what hardness assumption is being violated here? Depending on the problem formulation, the difficulty that ensures
unforgeability here comes from the _Ring Short Integer Solution_ problem. In particular, if I can extract
$\underline{x}^\prime$ and $\underline{y}^\prime$ such that $\langle \underline{a}, \underline{x}^\prime \rangle = \langle \underline{a},
\underline{x}\rangle$ and $\langle \underline{a}, \underline{y}^\prime \rangle = \langle \underline{a}, \underline{y}\rangle$, yet
$\underline{x}^\prime \neq \underline{x}$ and $\underline{y}^\prime \neq \underline{y}$, then I can play around with
some algebra to get a new key $\underline{t} \neq \underline{0}$ such that $\langle \underline{a}, \underline{t}\rangle
= 0$ and yet $\|t\|_\infty$ is still small... and finding a short solution to $\langle \underline{a},
\underline{t}\rangle = 0$ is precisely the Ring Short Integer Solution problem.

### Optimizing One-Time Lattice-Based Signatures

The security properties for lattice-based schemes often are based on requiring that public keys are covered by a
sufficiently dense set of private keys. This ensures that the short solution $\underline{t}$ to the Ring Short Integer
Solution problem is non-zero at least half the time. However, direct implementations of these schemes often leads to
unnecessarily dense key coverage.

For example, consider the requirement that a uniformly sampled element from the domain of the maps which carry the
secret keys $(\underline{x}, \underline{y})$ to the public keys and a signature $(X, Y, \xi)$ always has a distinct
second pre-image $(\underline{x}^\prime, \underline{y}^\prime)$. What sort of requirements does this place on the system
parameters?

Well, for any function $f: A \to B$, at most $|B|$ elements map uniquely under $f$.

![A surjective function](surjection.png)

Hence, if an element $a$ is sampled from $A$ uniformly at random, then there is a probability at most $|B|/|A|$ that $f(
a)$ has no other pre-images $a^\prime$. To ensure this probability is less than $2^{-\lambda}$, we require that $|B|
\cdot 2 ^ \lambda < |A|$.

The domain of our map is pairs of elements from $B(V_q, \beta_{sk})$. Thus, the size of the domain is the square of the
size of $B(V_q, \beta_{sk})$. Moreover, an element of $B(V_q, \beta_{sk})$ is an $\ell$-vector of polynomials whose
coefficients are absolutely bounded by $\beta_{sk}$... which means all coefficients are in the list
$[-\beta_{sk}, -\beta_{sk} + 1, ..., \beta_{sk} - 1, \beta_{sk}]$, which clearly has $2 \beta_{sk} + 1$ elements in it.
All such vectors are allowable, each has exactly $\ell$ coordinates, and each of those has exactly $d$ coefficients.
There are necessarily $(2\beta_{sk} + 1)^{\ell d}$ such vectors, so there are $(2\beta_{sk}+ 1)^{2\ell d}$ elements in
the domain of our map.

On the other hand, the codomain of our map is tuples containing pairs of public keys and signatures. Each public key is
a polynomial in $R_q$, and may be unbounded. So we have $q^{2d}$ possible public verification keys. On the other hand,
the signature is a polynomial vector with coefficients absolutely bounded by $\beta_{v}$. Hence, we have $(2 \beta_
{v}+1)^{\ell d}$ possible signatures. In particular, we have $q^{2d} \cdot (2\cdot\beta_{v} + 1)^{\ell d}$ elements in
the codomain.

So one requirement for security is that $\frac{(2\beta_{sk}+1)^{2\ell d}}{q^{2*d} (2\beta_{v} + 1)^{\ell d}} >
2^\lambda$.

Now, on the other hand, we may be able to use _sparse keys_ in order to make this inequality easier to satisfy. In
particular, we can consider a variation of the Schnorr-Like approach to the lattice setting wherein private keys are not
just polynomial vectors whose infinity norms are bounded by $\beta_{sk}$, but whose Hamming weights are also bounded by
some $1 \leq \omega_{sk} \leq d$. We can similarly put a bound on the Hamming weight of the signature challenge $c$ by some $1
\leq \omega_{ch} \leq d$, and we will consequently be bounding the Hamming weight of signatures by some $1 \leq \omega_
{v} \leq d$. In this case, our inequality constraint changes to become a significantly more complicated inequality
involving binomial coefficient computations, which we omit for the sake of readability. If we carefully select $\omega_
{sk}$, $\omega_{ch}$, and $\omega_{v}$, the above bound can be tightened or loosened. Using this technique, we can
describe signatures and keys with less space than otherwise. For example, if $\omega_{v} = d/8$ and $d = 512$, then we
can save at least $134$ bits per signature with efficient encoding of the signature. However, this choice may make the
inequality in question harder to satisfy (recall that the inequality in question guarantees second pre-images occur
often enough).

### Prototyping these lattice-based one-time signatures

In [[2]](https://github.com/geometry-labs/lattice-algebra), we present infrastructure for rapid implementation of these
lattice-based signatures; see [[3]](https://www.theqrl.org/blog/lattice-algebra-library/) for a description of this
package and the rationale behind it. In an upcoming python package, we support three distinct signature schemes based on
this approach:

1. LMSigs: A variation of the scheme in [[1]](https://eprint.iacr.org/2013/746.pdf) (and named for those authors) which
   uses one-time keys and one-time signatures with no further features. Because there are no further features, we can
   make parameters surprisingly small, to the point where our keys and signatures are competitively sized when compared
   to, e.g., CRYSTALS-Dilithium.
2. BKLMSigs: A variation of the scheme in [[4]](https://crypto.stanford.edu/~skim13/agg_ots.pdf) (and named for those
   authors as well as the authors of [[1]](https://eprint.iacr.org/2013/746.pdf)), which uses one-time keys and one-time
   signatures, but with non-interactive aggregation. In order to have support for an aggregation capacity, parameters
   are larger. Aggregation adds computational overhead to verification, which we cover in the next article in this
   series.
3. AdaSigs: Related to the schemes in [[5]](https://eprint.iacr.org/2020/845.pdf)
   and [[6]](https://eprint.iacr.org/2020/1345.pdf), this scheme uses one-time keys, one-time signatures, one-time
   witnesses, and one-time statements in order to support adaptor signatures. The computational overhead of AdaSigs is
   more similar to that of LMSigs than that of BKLMSigs. Due to the utility of adaptor signatures in flexible protocol
   design, and due to the efficiency comparisons between LMSigs, AdaSigs, and CRYSTALS-Dilithium, we think AdaSigs are a
   good choice for post-quantum protocol design.

### Transaction Protocols And Usual Digital Signatures

For the purposes of this discussion, a _transaction_ is a tuple ```(I, O, FEE, MEMO, AUTH)``` where ```I``` is a set
of _input keys_, ```O``` is a set of _output keys_, ```FEE``` is a positive plaintext fee, ```AUTH``` is an
authentication of some sort, and ```MEMO``` is an optional memo field. The transaction can be seen as consuming the
input keys and creating new output keys; in fact, every _input key_ in a transaction must be an _output key_ from an old
transaction, and whether a key is an input or an output key is therefore highly contextual. The outputs are injected
into the system via work or stake rewards, which allow users to authorize special negative-fee transactions. Users
broadcast these transactions on any network however they agree upon, validators assemble valid transactions into valid
blocks with whatever block puzzles they agree upon, and they order these blocks into a sensible linear transaction
history using whatever consensus mechanism they agree upon.

A transaction protocol can be built using a public bulletin board (upon which transactions are posted), together with
some sort of map or lookup called ```AMOUNT``` that maps outputs to plaintext transaction amounts. A transaction posted
on the bulletin board is considered _valid_ if every input reference points to a valid transaction, the signatures in
the authentication ```AUTH``` are valid, and ```AMOUNT(I) - AMOUNT(O) - FEE = 0```. Obviously this leads to a regression
of valid transactions, which must terminate at some sort of root transaction.

These root transactions are known as base transactions or coinbase transactions. Base transactions are special
transactions that (i) include a block puzzle solution in ```AUTH``` and (ii) have a deterministically
computed ```FEE < 0```, called the block reward. In a proof-of-work cryptocurrency, the block puzzle solution is often
the input to a hash function that produces a sufficiently unlikely output. In a proof-of-stake cryptocurrency, this
block puzzle solution is a signature by validators that have been decided upon in advance somehow.

From this point of view, then, to describe a transaction protocol is to (i) describe the structure
of ```(I, O, FEE, MEMO, AUTH)```, (ii) describe ```AMOUNT```, (iii) decide upon a block puzzle solution method for the
protocol, and (iv) decide upon a deterministically computed block reward.

Generally ```I``` can be lazily described using any unambiguous reference to a set of outputs from the transaction
history, such as a list of block height-transaction index-output index triples, and ```FEE``` is a plaintext amount in
every transaction. We can allow any bitstring memorandum message, but we prepend any user-selected memorandum with a
bitstring representation of the tuple ```(I, O, FEE)```. In practice, ```AUTH``` generally consists of a multi-signature
with the keys described in $I$ on the pre-pended memorandum message.

We need to stash into each output both a verification key from some unforgeable signature scheme and also some way of
measuring amounts. We can accomplish this with no privacy whatsoever by making an output key of the form $(vk, \alpha)$
where $\alpha$ is a plaintext amount and $vk$ is a public verification key from the underlying signature scheme. In
this case, the ```AMOUNT``` function merely forgets the verification key $vk$ and outputs $\alpha$.

To summarize, using an unforgeable signature scheme as a sub-scheme, we use the following structure of general
transactions for a transparent cryptocurrency.

1. The list of inputs ```I``` in a transaction consist of triples of the
   form ```(block_hash, transaction_id, output_index)```. Any validator with a block referenced by this transaction can
   easily look up an old output key from an old transaction in their ledger this way. This is assumed to be ordered in a
   canonical way.
2. The list of new outputs ```O``` consists of verification key-amount pairs $(vk, \alpha)$ where $vk$ is a verification
   key from the sub-scheme and $\alpha$ is a positive plaintext amount. This is assumed to be ordered in a canonical
   way. Thus, each ```(block_hash, transaction_id, output_index)``` refers to an output key $(vk, amt)$ in the
   validator's local copy of the blockchain.
3. The fee ```FEE``` is a user-selected positive amount.
4. The memo ```MEMO``` is a user-selected bitstring.
5. The authentication ```AUTH``` consists of a multi-signature on the message ```MSG = I || O || FEE || MEMO```
   where ```||``` denotes concatenation, signed by the inputs, i.e. the keys found at the input references.

Recalling that concatenating usual digital signatures together is a trivial way of making a multi-signature scheme, we
see that we can simply stack signatures to accomplish this protocol. This way, anyone with the secret signing keys $sk_
{old}$ associated to an output verification key $vk_{old}$ in some old valid transaction can construct a new valid
transaction sent to any target verification key $vk_{new}$ by simply signing the zero bit and setting the amounts.

However, this approach re-uses the verification keys of a user, so this approach does not carry over directly to
one-time signature schemes. This is easily rectified, as we see in the next section.

### Transaction Protocols And One-Time Signatures

We can use one-time signatures in a transaction protocol to introduce a degree of unlinkability, if we like, but we need
some sort of key exchange. This way, we can ensure that the recipient can extract the new one-time signing keys
associated with their new outputs, whilst ensuring that the sender cannot extract that key. We can introduce
auditor-friendly unlinkability by introducing one-time keys and using a CryptoNote-style view keys.

#### Example: Monero/CryptoNote-Style One-Time Keys

CryptoNote used the computational Diffie-Hellman assumption and hash functions in order to construct one-time keys from
many-time wallet keys. We show how this key exchange can be used to develop an amount-transparent and
non-sender-ambiguous version of CryptoNote that is nevertheless unlinkable and auditor-friendly.

In the following, let $(\mathbb{G}, p, G, F)$ be the group parameters from our Schnorr-like example of one-time
signatures.

0. In a setup phase, users agree upon $(\mathbb{G}, p, G, F)$, and another hash function $H:{0,1}^* \to
   \mathbb{Z}/p\mathbb{Z}$.
1. Users select many-time wallet keys by sampling $a, b \in \mathbb{Z}/p\mathbb{Z}$ independently and uniformly at
   random. Users set $A = aG$, $B = bG$, $sk = (a, b)$, and $vk = (A, B)$. Users then broadcast $vk$.
2. An output is a tuple $(R, P, \alpha)$ where $R$ and $P$ are group elements and $\alpha$ is a plaintext amount.
3. To send an amount $\alpha$ to another user with some $vk = (A, B)$, a new output $(R, P, \alpha)$ is constructed in
   the following way. First, we sample a new random $r \in \mathbb{Z}/p\mathbb{Z}$ independent of all previous samples
   and uniformly at random, and we set $R = rG$. Next, we use $r$ and the recipient $A$ to compute the Diffie-Hellman
   shared secret $rA = aR$. We hash this with $H$ and compute $P = H(rA)G + B$.
4. A user can easily determine if a new output $(R, P, \alpha)$ is addressed to their own $vk$ by checking if $P = H(aR)
   G + B$.
5. To send an old output $(R, P, \alpha)$, the amounts must balance and the sender must include a signature on the key
   $P$ in their authentication.

Under the Computational Diffie-Hellman assumption, only the recipient (owner of $a$) and the sender (who knows $r$) can
compute $rA = aR$, so only the sender and the recipient can link the output $(R, P, \alpha)$ with the wallet keys $(A,
B)$. This leads to one-time keys in the sense that only one signature ever needs to be posted in order for a transaction
to be considered valid and complete. However, depending on the properties of the underlying signature scheme, it may or
may not be safe to publish more than one signature from a key $P$. Moreover, the privacy gained from such an approach is
rather minimal, as the one-time addresses can easily be _probabilistically_ linked with one another by a studious third
party. Note that to determine whether a transaction is addressed to $(A, B)$, the recipient only need the private key
$a$, which is why we call it a _view key_. On the other hand, we call $b$ a _spend key_ because we need it to construct
new transactions.

The overall approach itself is a rather inelegant instantiation of a key encapsulation mechanism, and the one-time-ness
of the protocol can be a "soft" one-time-ness. By using a proper key encapsulation mechanism and one-time signatures, we
can make this a bit of a tighter ship.

#### A brief aside: Key Encapsulation Mechanisms

A key encapsulation mechanism (KEM) is a tuple of algorithms $(\texttt{Setup}, \texttt{Keygen}, \texttt{Enc},
\texttt{Dec})$ which work as follows.

0. $\texttt{Setup}(\lambda) \to \rho$ inputs a security parameter $\lambda$ and outputs some public parameters $\rho$.
1. $\texttt{Keygen} \to (sk, pk)$. Output a new secret-public keypair $(sk, pk)$.
2. $\texttt{Enc}(pk) \to (c, ek)$. Input a public key $pk$ and output a ciphertext $c$ and a bitstring $ek$ we call an _
   ephemeral key._
3. $\texttt{Dec}(sk, c) \to {ek, \bot}$. Inputs a secret key $sk$ and a ciphertext $c$, and outputs a bitstring $ek$ or
   a distinguished failure symbol $\bot.

Setting aside the security properties of a key encapsulation mechanism (that's a different article!), the rough idea
here is to compute $(c, ek) \leftarrow \texttt{Enc}(pk)$ and use $ek$ as a symmetric key to encrypt a message $\mu$,
resulting in another ciphertext $d$. You then send $(c, d)$ to the owner of $pk$, who can use their $sk$ to compute $ek
\leftarrow \texttt{Dec}(sk, c)$, and then use $ek$ to decrypt $d$ to obtain $\mu$.

#### Example: Extending to Lattice-Based One-Time Keys

In the following, let $(d, q, l, k, F, \beta_{sk}, \beta_{ch}, \beta_{v}, \underline{a})$ be output from $Setup$ in the
Schnorr-like approach to the lattice setting, let $\beta_t$ be some natural number, let $H_0, H_1:{0, 1}^* \to B(V_q,
\beta_t)$ and let $(KEM.Setup, KEM.Keygen, KEM.Enc, KEM.Dec)$ be a secure KEM.

0. In a setup phase, users agree upon all the above parameters, hash functions, and so on.
1. Users select many-time wallet keys $(sk, vk)$ in the following way. To sample $sk$, the user computes $(KEM.sk,
   KEM.pk) <- KEM.Keygen$, samples a new random $x_0, x_1$ uniformly at random from $B(V_q, \beta_{sk})$, computes $X_0
   = \langle \underline{a}_0, \underline{x}_0\rangle$, $X_1 = \langle \underline{a}_0, \underline{x}_1\rangle$ and sets
   $sk = (KEM.sk, x_0, x_1)$. They then use $vk = (KEM.vk, X_0, X_1)$.
2. An output is a tuple $(c, h_0, h_1, \alpha, \pi)$ where $c$ is a ciphertext from the KEM, $h$ and $g$ are
   polynomials, $\alpha \in \mathbb{N}$ is a plaintext amount, and $\pi$ is a zero-knowledge proof of knowledge (ZKPOK)
   that the sender knows some $\underline{z}_0, \underline{z}_1 \in B(V_q, \beta_{sk} + \beta_{t})$ such that $h_i =
   \langle \underline{a}, \underline{z}_i\rangle$ for $i=0, 1$.
3. An authorization to send an output is just a one-time signature using $(h_0, h_1)$ as the verification key.
4. To send an amount $\alpha$ to another user with some $vk = (KEM.pk, X_0, X_1)$, a new output $(c, h_0, h_1, \alpha,
   \pi)$ is constructed in the following way. First, we encapsulate a new ephemeral key for $KEM.pk$ by computing $(c,
   ek) <- KEM.Enc(KEM.pk)$. Next, we compute $\underline{b}_0, \underline{b}_1 = H_0(ek), H_1(ek)$ and $h_i = X_i +
   \langle \underline{a}, \underline{b}_i\rangle$ for $i=0, 1$. Next, we compute a ZKPOK $\pi$ that the signer knows
   elements $\underline{z}_i \in B(V_q, \beta_{sk} + \beta_t)$ such that $h_i = \langle \underline{a}, \underline{z}_
   i\rangle$ for $i=0, 1$ (namely $\underline{z}_i = x_i + b_i$). Now the sender can publish $(c, h_0, h_1, \alpha, \pi)
   $.
5. Upon hearing of a new transaction, a user can easily determine if a new output $(c, h_0, h_1, \alpha, \pi)$ is
   addressed to their own $vk$ by checking if $KEM.Dec(sk, c) != FAIL$, can check that the sender actually knows the
   pre-image of $h_0$ and $h_1$ by verifying $\pi$, and can manually check the plaintext amounts to verify that no money
   was created or destroyed. To obtain the signing keys corresponding to the public verification key $(h_0, h_1)$, the
   user decapsulates $c$ to obtain $ek^\prime$, computes $\underline{b}_i = H_i(ek)$, and then computes $\underline{z}_i
   = \underline{x}_i + \underline{b}_i$.

Note that for this scheme to be secure, we require all our hash functions to be cryptographically strong against finding
second pre-images, the KEM to be secure, and the signature scheme to be unforgeable.

### Conclusion

In the pursuit of more efficient lattice-based signing that is resistant to implementation mistakes, we investigate
one-time signatures. We discuss some simple signature designs, and their security properties, in the elliptic curve
setting to motivate discussion on these in the lattice-based settings. We describe a high-level view of how
lattice-based signatures can be made more efficient with sparse keys, and we link to a python package we developed that
employs these signatures. We describe how transaction protocols can be built from both usual and digital signature
schemes. Lastly, we describe a very simple implementation of a transaction protocol that uses Key Encapsulation
Mechanisms and one-time lattice-based signatures.

In the next part of this series, we explore the possibility of signature aggregation, wherein we try to represent
several signatures as an aggregate signature, and we discuss how to determine if signature aggregation is worth it. In
subsequent parts, we will discuss how to use adaptor signatures to lighten a blockchain's load via payment channels and
cross-chain atomic swaps. In the final part of the series, we look at using secure fountain architectures to minimize
local storage requirements without compromising blockchain security (at least up to a certain security model).

### References

[[1]](https://eprint.iacr.org/2013/746.pdf)  Lyubashevsky, Vadim, and Daniele Micciancio. "Asymptotically efficient
lattice-based digital signatures." Theory of Cryptography Conference. Springer, Berlin, Heidelberg, 2008.

[[2]](https://github.com/geometry-labs/lattice-algebra) Geometry Labs' lattice-algebra repository on GitHub.

[[3]](https://www.theqrl.org/blog/lattice-algebra-library/) TheQRL announcement of the lattice-algebra package.

[[4]](https://crypto.stanford.edu/~skim13/agg_ots.pdf) Boneh, Dan, and Sam Kim. "One-Time and Interactive Aggregate
Signatures from Lattices." (2020).

[[5]](https://eprint.iacr.org/2020/845.pdf) Esgin, Muhammed F., Oğuzhan Ersoy, and Zekeriya Erkin. "Post-quantum adaptor
signatures and payment channel networks." European Symposium on Research in Computer Security. Springer, Cham, 2020.

[[6]](https://eprint.iacr.org/2020/1345.pdf) Tairi, Erkan, Pedro Moreno-Sanchez, and Matteo Maffei. "Post-quantum
adaptor signature for privacy-preserving off-chain payments." International Conference on Financial Cryptography and
Data Security. Springer, Berlin, Heidelberg, 2021.