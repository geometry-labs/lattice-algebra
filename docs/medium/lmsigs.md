# Zen and the Art of Compact Post-Quantum Finance

Cryptographic schemes that are secure against quantum adversaries are infamous for their relative inefficiency in comparison to their classically secure counterparts. Nevertheless, the days are numbered for classical cryptography as quantum computers loom on the horizon. This naturally leads our interest toward cryptographic schemes and protocols that are designed for quantum resistance. Especially with the advent of cryptocurrencies, the race to develop robust and commercially viable quantum computers has a tantalizing and profitable prize awaiting the winners, and a highly disruptive economic future for the rest of us.  Hence, it is of critical importance to develop quantum-resistant cryptography tools and to begin migration to quantum-resistant settings as soon as possible, in the spirit of Quantum Resistant Ledger. This can mean sacrificing the efficiency of classical cryptography, but (of course!) a robust and secure cryptographic system is useless if it is too unwieldy to use. 

This is a multipart article series exploring avenues toward more efficient quantum-resistant transacting. Here in part one, we explore a candidate lattice-based one-time signature scheme. The scheme is rather similar in certain ways to the NIST post-quantum standard candidate, CRYSTALS-Dilithium, and provides an avenue to discuss optimizations available toward smaller keys and signatures in lattice-based cryptographic schemes. In part two, we cover the technical end of _signature aggregation_. The third part of this series explores _payment channels_ constructed with _adaptor signatures_, their security models, and their implementations. In the final part, we describe a blockchain-agnostic code-based approach to trustlessly (or, rather, _trust-minimally_) reducing local storage requirements for network participants while keeping bootstrapping costs low. Along the way, we present prototype python implementations of these ideas. 

## Part One: In Pursuit of More Efficient Lattice-Based Signing

### Digital Signatures

One-time signatures are useful for a variety of cryptographic applications, but to talk about one-time signatures, we first speak about usual digital signature schemes first.

A digital signature scheme is a tuple of algorithms ```(Setup, Keygen, Sign, Verify)``` which informally work as follows.
  0. ```Setup``` inputs a security parameter ```k``` and outputs some public parameters, ```pp```.
  1. ```Keygen``` inputs ```(k, pp)``` where ```k``` and ```pp``` are the input/output pair from ```Setup```, and outputs a new random keypair ```(sk, vk)``` where ```sk``` is a secret signing key and ```vk``` is a public verification key.
  2. ```Sign``` inputs ```(k, pp, (sk, vk), m)``` where ```k``` and ```pp``` are the input/output pair from ```Setup```, ```(sk, vk)``` is a key pair, and ```m``` is a bitstring. ```Sign``` then outputs a signature, ```sig```.
  3. ```Verify``` inputs ```(k, pp, vk, m, sig)``` where ```k``` and ```pp``` are the input/output pair from ```Setup```, ```vk``` is a public verification key, ```m``` is a bitstring, and ```sig``` is a purported message, and then outputs a bit ```b``` indicating whether the signature is a valid signature on the message ```m``` with the public verification key ```vk```.

We can symbolically represent the above digital signature scheme as follows, by taking ```k, pp``` for granted in each algorithm.
  1. ```Keygen -> (sk, vk)```
  2. ```Sign((sk, vk), m) -> sig```.
  3. ```Verify(vk, m, sig) -> b```.
  
Such a scheme, as described, is rather useless, without further conditions. This leads us to the notion of the security properties of digital signatures.

### Security Properties: Correctness

Loosely, we say a signature scheme is _correct_ whenever, for any message ```m```, for any pair ```(sk, vk)``` that is output from ```Keygen```,  ```Verify(vk, m, Sign((sk, vk), m)) = 1```. Correctness here is in analogy to soundness in zero-knowledge protocols. When considering security properties, it is often helpful to ask "what does it mean for a scheme to satisfy the _negation_ of this property?"

For example, the negation of the correctness definition is as follows: there exists a message ```m``` and a keypair ```(sk, vk)``` that is output from ```Keygen``` such that ```Verify(vk, m, Sign((sk, vk), m)) = 0```. In this case, a scheme that is not correct is not guaranteed to even produce signatures that pass verification!

### Security Properties: Unforgeability

Now consider what it means for a signature scheme to be _secure_; more precisely, consider the notion of unforgeability. If Alice attempts to forge a usual digital signature without knowing Bob's key, we can formalize this situation in a three-step process. First, Alice and Bob agree upon the input security parameter ```k``` for ```Setup```, and they can each compute ```Setup(k) -> pp``` themselves; this models the real-life situation where ```k``` and ```pp``` are agreed-upon parameters that have been publicly audited and cryptanalyzed (cf. the NIST post-quantum signature vetting process). Next, Bob runs ```Keygen```, obtains the _challenge keys_ ```(sk, vk)```, and sends ```vk``` to Alice. Lastly, Alice attempts to output some message-signature pair ```(m, sig)```. Alice's forgery is successful if ```Verify(vk, m, sig) == 1```.

The idea behind unforgeability is that any algorithm "Alice" should fail at this game except with negligible probability.

There are two main things to note about this game. First, of course, the game is easy if Bob gives away his ```sk``` to Alice, so it is critical that Alice learns no information about ```sk``` in the course of the game in order for the result to be considered a forgery. Second, the game fails to capture the real-life situation where Alice may have seen signatures published under Bob's key before, perhaps posted on some public bulletin board. Thus, Alice is often granted the benefit of the doubt and is given access to a _signature oracle_. This allows her to obtain a signature from Bob's key on any message she likes... this models the real-life situation where Alice may be able to coerce or trick Bob into signing a message that Alice has selected. However, in this case, Alice certainly should not be allowed to win the game by passing off an oracle-generated signature as a forgery. Moreover, Alice generally only cares about forging signatures _for which she does not already have a signature._ Hence, Alice is generally not interested in winning the game even by re-using any of the messages that have already been signed by the oracle.

This leads us to the following description of the _existential unforgeability experiment_. First, Alice and Bob agree upon ```k``` and compute ```pp```. Next, Bob generates ```(sk, vk) <- Keygen``` and sends ```vk``` to Alice. Next, Bob grants Alice signature oracle access where she can query the oracle with any message ```m``` and obtain a signature ```sig``` such that ```Verify(vk, m, sig) == 1```. Eventually, Alice outputs a message-signature pair ```(m, sig)```, succeeding if and only if the signing oracle was not queried with ```m``` and ```Verify(vk, m, sig) == 1```.

We then define a scheme to be _existentially unforgeable_ if, for any algorithm "Alice", the probability that Alice succeeds at this experiment is negligible.

As before, let us consider what it means to negate the definition of existential unforgeability. Then there exists an algorithm "Alice" that has a non-negligible probability of winning this game. This means that Alice, with no information about Bob's ```sk``` but perhaps with the ability to persuade Bob to publish some signatures on messages of Alice's choice, can nevertheless produce a new forgery on a message Bob has never seen before.

Of course, we can expand this game to include more challenge keys; Bob could say "here are several of my verification keys, I challenge you to construct a valid signature on any one of them." However, the number of keys provided to Alice is always polynomially bounded in ```k```, and thus Alice can at best polynomially improve her probability of success. In particular, if we can assume that Alice succeeds with only negligible probability in the one-key case, then Alice still can only succeed with negligible probability in the case that she is given a polynomial number of challenge keys, so we don't need to consider this generalization.

### Variations: One-Time Signature Schemes

A one-time signature scheme is just a digital signature scheme, even with the same definition of correctness, but with a different definition of unforgeability. In one-time schemes, producing more than one signature from the same key can reveal that key to other users, so it is only ever safe to publish one signature from any key; this makes one-time signatures particularly useful in constructing transaction protocols, as we shall see. However, this also means that our definition of existential unforgeability is no longer suitable. After all, in that definition, Alice can request many signatures from the signing oracle. To rectify this, the definition of the _one-time existential unforgeability experiment_ is nearly identical to the existential unforgeability experiment, except Alice is only granted _one_ oracle query.

### Simplicity First: “Clever” Is Not a Compliment in Cryptography

One-time signature schemes are oftentimes very simple to describe and implement, which leads to straightforward proofs of security properties and mistake-resistant code. In the following examples, we show a case from classically secure cryptography and a similar case from quantum-resistant cryptography. As we can see from the descriptions below, describing the schemes is quite simple. Thus, implementing these schemes has fewer pitfalls than implementing more "clever" signature schemes. 

#### Example: A Schnorr-Like One-Time Signature Scheme

Let ```g``` be a generator of an elliptic curve group ```G``` of order ```p``` over which the discrete logarithm game is hard, and assume ```g``` is a group element sampled uniformly at random. Let ```F``` be a random oracle function that outputs elements from the integers modulo ```p```. The following defines a one-time signature scheme that satisfies one-time existential unforgeability.

  0. ```Setup(k) -> pp```. Set ```pp = (G, p, g, F)```
  1. ```Keygen -> (sk, vk)```. Sample ```r, s``` independently and uniformly at random from the non-zero integers modulo ```p```, set ```x = g ** r, y = g ** s```. Set ```sk = (r, s)``` and ```vk = (x, y)```. Output ```(sk, vk)```.
  2. ```Sign((sk, vk), m) -> sig```. Compute ```c = F(vk, m)```, parse ```(r, s) = sk```, set ```sig = (r * c + s) % p```, and output ```sig```.
  3. ```Verify(vk, m, sig) -> b```. Check that ```sig``` is an integer modulo ```p```, compute ```c = F(vk, m)```, and output ```1``` if and only if ```g ** sig == x ** F(vk, m) * y```.

The scheme is correct. Indeed, ```sig``` is an integer modulo ```p``` since ```r```, ```c```, and ```s``` are integers modulo ```p``` and ```sig``` is a linear combination of these (modulo ```p```). Also, since ```Sign((sk, vk), m) = r * c + s```, we have that ```g ** (r * c + s) = (g** r)** c * g** s = x ** c * y```. The scheme can also be shown to be unforgeable, by showing that if Alice can produce a forgery, then Alice can also break the discrete logarithm assumption. Indeed, if Alice can produce ```sig``` that satisfies the game, and receives an oracle-generated signature ```o_sig```, then Alice can compute ```sig - o_sig``` easily and solve backward for ```r``` and ```s```. Hence, if Alice sees a discrete logarithm challenge  ```x```, she can roll her own ```s```, compute ```y = g ** s```, and then play the unforgeability game with the challenge keys ```(x, y)```. At the end, she will have successfully gained enough information to easily compute the discrete logarithm of ```x```. Since computing discrete logarithms is hard with classical computers, we conclude that classical forgers do not exist (or succeed with very low probability, or take a very long time to finish their tasks).

#### Example: Extend this Schnorr-Like Approach to the Lattice Setting

We can swap out the fundamental hardness assumption above with a different hardness assumption to get a different scheme; using a related approach, Lyubashevsky and Micciancio first proposed signatures designed (mostly) this way in [[1]](https://eprint.iacr.org/2013/746.pdf). In a lattice setting, we ditch the elliptic curve group in favor of lattices induced by rings of integers and their ideals. More precisely, let ```Z``` be short for the integers, let ```Zq``` be short for the integers modulo ```q``` for a prime modulus ```q```, let ```d``` be a power of two integer, let ```l``` be a natural number, let ```R = Z[X]/(X ** d + 1)``` be a quotient ring of integer-coefficient polynomials, let ```Rq``` be the quotient ring ```R/qR```, let ```V = R ** l``` be a module over ```R``` of length ```l```, and let ```Vq = Rq ** l``` be a module over ```Rq``` with length ```l```. Note that ```V``` and ```Vq``` are also vector spaces over ```Zq``` of dimension ```d*l```. Elements of ```R``` have an infinity norm, which is the absolute maximum of their coefficients, which we denote with ```inf_norm```. Extend the infinity norm to ```Rq```, ```V```, and ```Vq``` as usual.  Let ```sk_bd```, ```ch_bd```, and ```vf_bd``` be natural numbers.  Abuse notation and define ```R(bd)```, ```Rq(bd)```, ```V(bd)```, and ```Vq(bd)``` as the subsets of ```R```, ```Rq```, ```V``` and ```Vq```, respectively, consisting of elements ```t``` such that ```inf_norm(t) <= bd```. Unlike before, let ```F``` be a random oracle whose codomain is ```Rq(ch_bd)```.  

  0. ```Setup(k) -> pp```. Sample ```a``` from ```Vq``` uniformly at random and set ```pp = (d, q, l, F, sk_bd, ch_bd, vf_bd, a)``` (or just ```pp = (V, F, sk_bd, ch_bd, vf_bd, a)```).
  1. ```Keygen -> (sk, vk)```. Sample ```r, s``` uniformly at random and independently from ```Vq(sk_bd)```. Define ```x = a * r``` and ```y = a * s``` where ```*``` here denotes the dot product between two vectors.
  2. ```Sign((sk, vk), m) -> sig```. Compute ```c = F(vk, m)```, set ```sig = r * c + s``` where ```*``` here denotes scaling the polynomial vector ```r``` from ```Vq(sk_bd)``` with the polynomial ```c``` from ```Rq(ch_bd)```.
  3. ```Verify(vk, m, sig) -> b```. Check that ```sig``` is an element of ```Vq(vf_bd)```, compute ```c = F(vk, m)```, and output ```1``` if and only if ```a * sig == x * c + s``` where the ```*``` on the left-hand side is a dot product and the ```*``` on the right-hand side is simple polynomial multiplication.

This scheme may not always be correct, depending on how norms grow in ```R```, although correctness is easily attained. In particular, we note that since ```sig = r * c + s```, correctness requires that ```vf_bd >= inf_norm(r * c + s)``` for every ```r```, ```c```, and ```s```. Moreover, bounds on ```r * c + s``` are rather straightforward to compute, so correctness can easily be attained.

Similarly to the previous scheme, this scheme can be proven unforgeable. Indeed, just like in the previous example, if Alice can produce a forgery, she can use this forgery together with the oracle-generated signature she receives in order to extract some keys ```r'``` and ```s'```. Note the apostrophes! In general, we can't necessarily extract the keys exactly, but we can extract _some keys_ that have matching public keys! 

But what hardness assumption is being violated here? Depending on the problem formulation, the difficulty that ensures unforgeability here comes from the _Ring Short Integer Solution_ problem. In particular, if I can extract ```r'``` and ```s'``` such that ``` a * r' = a * r```, ```a * s' = a * s```, and yet ```r' != r``` and ```s' != s```, then I can play around with some algebra to get a new key ```t != 0``` such that ``` a * t = 0``` and yet ```inf_norm(t)``` is still small... and finding a short solution to ```a * t = 0``` is precisely the Ring Short Integer Solution problem.

### Optimizing One-Time Lattice-Based Signatures

The security properties for lattice-based schemes often are based on requiring that public keys are covered by a sufficiently dense set of private keys. This ensures that the short solution ```t``` to the Ring Short Integer Solution problem is non-zero. However, direct implementations of these schemes often leads to unnecessarily dense key coverage.

For example, consider the requirement that a uniformly sampled element from the domain of the maps which carry the secret keys ```(r, s)``` to the public keys and a signature ```(x, y, sig)``` always has a distinct second pre-image ```(r', s')```. How can we ask what sort of requirements this places on the system parameters?

Well, for any function ```f: X -> Y```, at most ```Y``` elements map uniquely under ```f```. Hence, if an element ```r``` is sampled from ```X``` uniformly at random, then there is a probability at most ```|Y|/|X|``` that ```f(r)``` has no other pre-images ```r'```. To ensure this probability is negligible in the security parameter ```k```, we require only that ```|Y| * 2 ** k < |X|```.

The domain of our map is pairs of elements from ```Vq(sk_bd)```. Thus, the size of the domain is the square of the size of ```Vq(sk_bd)```. Moreover, an element of ```Vq(sk_bd)``` is an ```l```-vector of polynomials whose coefficients are absolutely bounded by ```sk_bd```... which means all coefficients are in the list ```[-sk_bd, -sk_bd + 1, ..., sk_bd - 1, sk_bd]```, which clearly has ```2 * sk_bd + 1``` elements in it. All such vectors are allowable, each has exactly ```l``` coordinates, and each of those has exactly ```d``` coefficients. There are necessarily ```(2*sk_bd + 1)**(l*d)``` such vectors, so there are ```(2*sk_bd + 1)**(2*l*d)``` elements in the domain of our map.

On the other hand, the codomain of our map is tuples containing pairs of public keys and signatures. Each public key is a polynomial in ```Rq```, and may be unbounded. So we have ```q ** (2*d)``` possible public verification keys. On the other hand, the signature is a polynomial vector with coefficients absolutely bounded by ```vf_bd```. Hence, we have ```(2*vf_bd+1)**(l*d)``` possible signatures. In particular, we have ```q ** (2*d) * (2*vf_bd + 1) **(l*d)``` elements in the codomain.

So one requirement for security is that ```(2*sk_bd+1)**(2*l*d) / (q**(2*d) *(2*vf_bd + 1)**(l*d)) > 2 ** k```.

Now, on the other hand, we may be able to use _sparse keys_ in order to make this inequality easier to satisfy. In particular, we can consider a variation of the Schnorr-Like approach to the lattice setting wherein private keys are not just polynomial vectors whose infinity norms are bounded by ```sk_bd```, but whose Hamming weights are also bounded by some ```1 <= sk_wt <= d```. We can similarly bound the Hamming weight of the signature challenge ```c``` by some ```1 <= ch_wt <= d```, and we will consequently be bounding the Hamming weight of signatures by some ```1 <= vf_wt <= d```. In this case, our inequality constraint changes to become a significantly more complicated inequality involving binomial coefficient computations, which we omit for the sake of readability.

If we carefully select ```sk_wt```, ```ch_wt```, and ```vf_wt```, the above bound can be tightened or loosened. For instance, by ensuring that ```vf_wt = d``` and making ```2 <= sk_wt <= d/2``` , the number of signing keys that could be responsible for a verification key and signature becomes significantly larger than otherwise, allowing our inequality to be satisfied with smaller values of ```sk_bd``` without changing ```d```, ```q```, ```l```, or ```vf_bd```. 

On the flip side, by selecting these parameters carefully, we can describe signatures and keys with less space than otherwise. For example, if ```vf_wt = d/8``` and ```d = 512```, then we can save at least ```134``` bits per signature with efficient encoding of the signature. However, this choice may make the inequality in question harder to satisfy (recall that the inequality in question guarantees second pre-images occur often enough).  

### Prototyping these lattice-based one-time signatures

In [[2]](https://github.com/geometry-labs/lattice-algebra), we present infrastructure for rapid implementation of these lattice-based signatures; see [[3]](https://www.theqrl.org/blog/lattice-algebra-library/) for a description of this package and the rationale behind it. In an upcoming python package, we support three distinct signature schemes based on this approach:
  1. LMSigs: A variation of the scheme in [[1]](https://eprint.iacr.org/2013/746.pdf) (and named for those authors) which uses one-time keys and one-time signatures with no further features. Because there are no further features, we can make parameters surprisingly small, to the point where our keys and signatures are competitively sized when compared to, e.g., CRYSTALS-Dilithium.
  2. BKLMSigs: A variation of the scheme in [[4]](https://crypto.stanford.edu/~skim13/agg_ots.pdf) (and named for those authors as well as the authors of [[1]](https://eprint.iacr.org/2013/746.pdf)), which uses one-time keys and one-time signatures, but with non-interactive aggregation. In order to have support for an aggregation capacity, parameters are larger. Aggregation adds computational overhead to verification, which we cover in the next article in this series.
  3. AdaSigs: Related to the schemes in [[5]](https://eprint.iacr.org/2020/845.pdf) and [[6]](https://eprint.iacr.org/2020/1345.pdf), this scheme uses one-time keys, one-time signatures, one-time witnesses, and one-time statements in order to support adaptor signatures. The computational overhead of AdaSigs is more similar to that of LMSigs than that of BKLMSigs. Due to the utility of adaptor signatures in flexible protocol design, and due to the efficiency comparisons between LMSigs, AdaSigs, and CRYSTALS-Dilithium, we think AdaSigs are a good choice for post-quantum protocol design.

### Transaction Protocols And Usual Digital Signatures

For the purposes of this discussion, a _transaction_ is a tuple ```(I, O, FEE, MEMO, AUTH)]``` where ```I``` is a set of _input keys_, ```O``` is a set of _output keys_, ```FEE``` is a positive plaintext fee, ```AUTH``` is an authentication of some sort, and ```MEMO``` is an optional memo field.  The transaction can be seen as consuming the input keys and creating new output keys; in fact, every _input key_ in a transaction must be an _output key_ from an old transaction, and whether a key is an input or an output key is therefore highly contextual. Users broadcast these transactions on any network however they agree upon, validators assemble valid transactions into valid blocks with whatever block puzzles they agree upon, and they order these blocks into a sensible linear transaction history using whatever consensus mechanism they agree upon.

A transaction protocol can be built using a public bulletin board (upon which transactions are posted), together with some sort of map or lookup called ```AMOUNT``` that maps outputs to plaintext transaction amounts. A transaction posted on the bulletin board is considered _valid_ if every input reference points to a valid transaction, the signatures in the authentication ```AUTH``` are valid, and ```AMOUNT(I) - AMOUNT(O) - FEE = 0```. Obviously this leads to a regression of valid transactions, which must terminate at some sort of root transaction. 

These root transactions are known as base transactions or coinbase transactions. Base transactions are special transactions that (i) include a block puzzle solution in ```AUTH``` and (ii) have a deterministically computed ```FEE < 0```, called the block reward. In a proof-of-work cryptocurrency, the block puzzle solution is often the input to a hash function that produces a sufficiently unlikely output. In a proof-of-stake cryptocurrency, this block puzzle solution is a signature by validators that have been decided upon in advance somehow.

From this point of view, then, to describe a transaction protocol is to (i) describe the structure of ```(I, O, FEE, MEMO, AUTH)]```, (ii) describe ```AMOUNT```, (iii) decide upon a block puzzle solution method for the protocol, and (iv) decide upon a deterministically computed block reward.

Generally ```I``` can be lazily described using any unambiguous reference to a set of outputs from the transaction history, such as a list of block height-transaction index-output index triples, and ```FEE``` is a plaintext amount in every transaction. We can allow any bitstring memorandum message, but we prepend any user-selected memorandum with a bitstring representation of the tuple ```(I, O, FEE)```. In practice, ```AUTH``` generally consists of a multi-signature with the keys described in ```I``` on the pre-pended memorandum message.

We need to stash into each output both a verification key from some unforgeable signature scheme and also some way of measuring amounts. We can accomplish this with no privacy whatsoever by making an output key of the form ```(vk, amt)``` where ```amt``` is some plaintext amount and ```vk``` is a public verification key from the underlying signature scheme. In this case, the ```AMOUNT``` function merely forgets the verification key and outputs ```amt```.

To summarize, using an unforgeable signature scheme as a sub-scheme, we use the following structure of general transactions for a transparent cryptocurrency. 
 1. The list of inputs ```I``` in a transaction consist of triples of the form ```(block_hash, transaction_id, output_index)```. Any validator with a block referenced by this transaction can easily look up an old output key from an old transaction in their ledger this way. This is assumed to be ordered in a canonical way.
 2. The list of new outputs ```O``` consists of verification key-amount pairs ```(vk, amt)``` where ```vk``` is a verification key from the sub-scheme and ```amt``` is a positive plaintext amount. This is assumed to be ordered in a canonical way. Thus, each ```(block_hash, transaction_id, output_index)``` refers to some output key ```(vk, amt)``` in the validator's local copy of the blockchain.
 3. The fee ```FEE``` is a user-selected positive amount.
 4. The memo ```MEMO``` is a user-selected bitstring.
 5. The authentication ```AUTH``` consists of a multi-signature on the message ```MSG = I || O || FEE || MEMO``` where ```||``` denotes concatenation, signed by the inputs, i.e. the keys found at the input references.

Recalling that concatenating usual digital signatures together is a trivial way of making a multi-signature scheme, we see that we can simply stack signatures to accomplish this protocol.  

This way, anyone with the secret signing keys ```sk_old``` associated to an output verification key ```vk_old``` in some old valid transaction can construct a new valid transaction sent to any target verification key ```vk_new``` by simply signing the zero bit and setting the amounts.

However, this approach re-uses the verification keys of a user, so this approach does not carry over directly to one-time signature schemes. This is easily rectified, as we see in the next section.

### Transaction Protocols And One-Time Signatures

We can use one-time signatures in a transaction protocol to introduce a degree of unlinkability, if we like, but we need some sort of key exchange. This way, we can ensure that the recipient can extract the new one-time signing keys associated with their new outputs, whilst ensuring that the sender cannot extract that key. We can introduce auditor-friendly unlinkability by introducing one-time keys and using a CryptoNote-style view keys. 

#### Example: Monero/CryptoNote-Style One-Time Keys

CryptoNote used the computational Diffie-Hellman assumption and hash functions in order to construct one-time keys from many-time wallet keys. We show how this key exchange can be used to develop an amount-transparent and non-sender-ambiguous version of CryptoNote that is nevertheless unlinkable and auditor-friendly.

In the following, let ```(G, p, g, F)``` be the group parameters from our Schnorr-like example of one-time signatures.

  0. In a setup phase, users agree upon ```(G, p, g, F)```, and another hash function ```H``` with codomain is the integers modulo ```p```.
  1. Users select many-time wallet keys by sampling ```(a, b)``` independently and uniformly at random from the integers modulo ```p```. Users set ```A = g**a```, ```B = g**b```, ```sk = (a, b)```, and ```vk = (A, B)```. Users then broadcast ```vk```.
  2. An output is a tuple ```(R, P, amt)``` where ```R``` and ```P``` are group elements and ```amt``` is a plaintext amount.
  3. To send an amount ```amt``` to another user with some ```vk = (A, B)```, a new output ```(R, P, amt)``` is constructed in the following way. First, we sample a new random ```r``` independent of all previous samples and uniformly at random from the integers modulo ```p```, and we set ```R = g**r```. Next, we use ```r``` and the recipient ```A``` to compute the Diffie-Hellman shared secret ```A ** r = R ** a```. We hash this with ```H``` and compute ```P = g ** H(A ** r) * B```. 
  4. A user can easily determine if a new output ```(R, P, amt)``` is addressed to their own ```vk``` by checking if ```P == g ** H(R ** a) * B```.
  5. To send an old output ```(R, P, amt)```, the amounts must balance and the sender must include a signature on the key ```P``` in their authentication.

Under the Computational Diffie-Hellman assumption, only the recipient (owner of ```a```) and the sender (who knows ```r```) can compute ```A ** r = R ** a```, so only the sender and the recipient can link the output ```(R, P, amt)``` with the wallet keys ```(A, B)```. This leads to one-time keys in the sense that only one signature ever needs to be posted in order for a transaction to be considered valid and complete. However, depending on the properties of the underlying signature scheme, it may or may not be safe to publish more than one signature from a key ```P```. Note that to determine whether a transaction is addressed to ```(A, B)```, the recipient only need the private key ```a```, which is why we call it a _view key_. On the other hand, we call ```b``` a _spend key_ because we need it to construct new transactions. 

The overall approach itself is a rather inelegant instantiation of a key encapsulation mechanism, and the one-time-ness of the protocol can be a "soft" one-time-ness. By using a proper key encapsulation mechanism and one-time signatures, we can make this into a proper one-time protocol.

#### A brief aside: Key Encapsulation Mechanisms

A key encapsulation mechanism (KEM) is a tuple of algorithms ```(Setup, Keygen, Enc, Dec)``` which work as follows.
  0. ```Setup``` inputs a security parameter ```k``` and outputs some public parameters ```pp```.
  1. ```Keygen``` inputs ```(k, pp)``` and outputs a new secret-public keypair ```(sk, pk)```.
  2. ```Enc``` inputs ```(k, pp)``` and a public key ```pk```, and outputs a ciphtertext ```c``` and bitstring ```ek```.
  3. ```Dec``` inputs ```(k, pp)```, a secret key ```sk```, and a ciphertext ```c```, and outputs a bitstring ```ek``` or a distinguished failure symbol FAIL.

Setting aside the security properties of a key encapsulation mechanism (that's a different article!), the idea is to send a secret ephemeral key ```ek``` to a recipient that can be used in a traditional key exchange to compute keys in a verifiable way. We provide an example in the next section. 

#### Example: Extending to Lattice-Based One-Time Keys

In the following, let ```(d, q, l, F, sk_bd, ch_bd, vf_bd, a)``` be output from ```Setup``` in the Schnorr-like approach to the lattice setting, let ```txn_bd``` be some natural number, let ```H``` be a hash function whose codomain is ```Vq(txn_bd)```, and let ```(KEM.Setup, KEM.Keygen, KEM.Enc, KEM.Dec)``` be a secure KEM.

  0. In a setup phase, users agree upon all the above parameters, hash functions, and so on.
  1. Users select many-time wallet keys ```(sk, vk)``` in the following way. To sample ```sk```, the user computes ```(KEM.sk, KEM.pk) <- KEM.Keygen```, samples a new random ```x``` uniformly at random from ```Vq(sk_bd)```, computes ```y = a * x``` where this ```*``` indicates the dot product of vectors, and they set ```sk = (KEM.sk, x)```. They then use ```vk = (KEM.vk, y)```.
  2. An output is a tuple ```(c, h, amt)``` where ```c``` is a ciphertext from the KEM and ```h``` is a polynomial.
  3. To send an amount ```amt``` to another user with some ```vk = (KEM.pk, y)```, a new output ```(c, h, amt)``` is constructed in the following way. First, we encapsulate a new ephemeral key for ```KEM.pk``` by computing ```(c, ek) <- KEM.Enc(KEM.pk)```. Next, we compute ```b = H(ek)``` and ```h = y + a * b``` where this ```*``` indicates the dot product of vectors. Now the sender can publish ```(c, h, amt)```.   
  4. A user can easily determine if a new output ```(c, h, amt)``` is addressed to their own ```vk``` by checking if ```KEM.Dec(sk, c) != FAIL```.
  5. To send an old output ```(c, h, amt)```, the amounts must balance and the sender must include a signature on the key ```h``` in their authentication.

Note that for this scheme to be secure against forgery, it is necessary (although not necessarily sufficient!) that publishing a signature on the key ```h = y + a * b = a * x + a * b``` is difficult without knowledge of both ```x``` and ```b```.

### Conclusion

In the pursuit of more efficient lattice-based signing that is resistant to implementation mistakes, we investigate one-time signatures. We discuss some simple signature designs, and their security properties, in the elliptic curve setting to motivate discussion on these in the lattice-based settings. We describe a high-level view of how lattice-based signatures can be made more efficient with sparse keys, and we link to a python package we developed that employs these signatures. We describe how transaction protocols can be built from both usual and digital signature schemes. Lastly, we describe a very simple implementation of a transaction protocol that uses Key Encapsulation Mechanisms and one-time lattice-based signatures.

In the next part of this series, we explore the possibility of signature aggregation, wherein we try to represent several signatures as an aggregate signature, and we discuss how to determine if signature aggregation is worth it. In subsequent parts, we will discuss how to use adaptor signatures to lighten a blockchain's load via payment channels and cross-chain atomic swaps. In the final part of the series, we look at using secure fountain architectures to minimize local storage requirements without compromising blockchain security (at least up to a certain security model).

### References

[[1]](https://eprint.iacr.org/2013/746.pdf)  Lyubashevsky, Vadim, and Daniele Micciancio. "Asymptotically efficient lattice-based digital signatures." Theory of Cryptography Conference. Springer, Berlin, Heidelberg, 2008.
[[2]](https://github.com/geometry-labs/lattice-algebra) Geometry Labs' lattice-algebra repository on GitHub.
[[3]](https://www.theqrl.org/blog/lattice-algebra-library/) TheQRL announcement of the lattice-algebra package.
[[4]](https://crypto.stanford.edu/~skim13/agg_ots.pdf) Boneh, Dan, and Sam Kim. "One-Time and Interactive Aggregate Signatures from Lattices." (2020).
[[5]](https://eprint.iacr.org/2020/845.pdf) Esgin, Muhammed F., Oğuzhan Ersoy, and Zekeriya Erkin. "Post-quantum adaptor signatures and payment channel networks." European Symposium on Research in Computer Security. Springer, Cham, 2020.
[[6]](https://eprint.iacr.org/2020/1345.pdf) Tairi, Erkan, Pedro Moreno-Sanchez, and Matteo Maffei. "Post-quantum adaptor signature for privacy-preserving off-chain payments." International Conference on Financial Cryptography and Data Security. Springer, Berlin, Heidelberg, 2021.