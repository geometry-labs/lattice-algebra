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

This is a multipart article series exploring avenues toward more efficient quantum-resistant transacting. Back in part
one, we explored a candidate lattice-based one-time signature scheme. The scheme is rather similar in certain ways to
the NIST post-quantum standard candidate, CRYSTALS-Dilithium, and provides an avenue to discuss optimizations available
toward smaller keys and signatures in lattice-based cryptographic schemes. In the second part, we covered the technical
end of _signature aggregation_, at least from one angle. Here in part three, we explore applications built from _adaptor
signatures_, the security models of adaptor signatures, and their implementations. In the final part, we describe a
blockchain-agnostic code-based approach to trustlessly (or, rather, _trust-minimally_) reducing local storage
requirements for network participants while keeping bootstrapping costs low.

## Part Three: Lightening the Load

### Cross-Chain and Off-Chain Transacting

A popular way of looking at blockchains is that they are slow and expensive base layers upon which lighter architecture
can be built. Storage and verification are costs paid collectively, so there is a high priority placed upon minimizing
the total number of interactions with this expensive base layer.

One way users can lighten storage requirements for nodes is to allow users to put collateral up in _payment channels_.
This way, users can transact off-chain, only settling to the base layer occasionally. This is the so-called "lightning
network" model of collateralized swaps. In a slight abuse of prefixes, we can think of these methods as "intra-coin"
collateralized swaps. The usage of this terminology is evocative of an "inter-coin" collateralized swap, which would
allow transacting between two chains.

As it turns out, we can accomplish both of these styles of collateralized swap using so-called _adaptor signatures_,
which have also been known as _verifiably encrypted signatures_.

### How Do Adaptor Signatures Work, and What Are They?

Adaptor signature schemes are cryptographic schemes that have all the functionality of usual digital signature schemes,
but with additional functionality: an adaptor signature scheme produces not just signatures, but also authenticated
commitments called _pre-signatures_. These pre-signatures commit to secret witnesses, can be verified to have been
computed by a signing key, and can be adapted into valid signatures given the secret witness. However, when pre-
signatures are adapted, the commitment is opened, revealing the secret witness. In fact, we require as a subscheme a _
hard relation_. Recall that a relation on a pair of sets $S, T$ is a subset of $S \times T$.

1. $Setup^*(\lambda) \to \rho$ is a setup algorithm that specifies a witness set $K_W$ and a statement set $K_T$.
2. $RelationGen \to (\underline{w}, s) \in K_W \times K_T$ generates a new witness-statement pair.
3. $RelationVerify(\underline{w}, s) \to b \in \left\{0, 1\right\}^*$ inputs a witness-statement pair and outputs a bit
   indicating whether the witness is valid for the statement.

We say this relation is a hard relation when finding a witness $\underline{w}$ given some $s$ is difficult. We may have
a nested pair of hard relations, say $K_W \subseteq K_W^\prime$ and $K_T \subseteq K_T^\prime$, such that
a pair from the "inner" relation is also a pair from the outer relation. We decorate data with the apostrophe to
indicate that the data come from the relaxed relation.

1. $Setup^*(\lambda) \to \rho$ is a setup algorithm that specifies witness sets $K_W \subseteq K_W^\prime$ and  
   statement sets $K_T \subseteq K_T^\prime$.
2. $RelationGen \to (\underline{w}, s) \in K_W \times K_T$ generates a new witness-statement pair.
3. $RelationGen^\prime \to (\underline{w}^\prime, s^\prime) \in K_W^\prime \times K_T^\prime$ generates a new
   relaxed witness-statement pair.
4. $RelationVerify(\underline{w}, s) \to b \in \left\{0, 1\right\}^*$ inputs a witness-statement pair and outputs a bit
   indicating whether the pair $(\underline{w}, s) \in K_W \times K_T$ is in the relation.
5. $RelationVerify^\prime(\underline{w}, s) \to b \in \left\{0, 1\right\}^*$ inputs a witness-statement pair and outputs
   a bit indicating whether the pair $(\underline{w}^\prime, s^\prime) \in K_W^\prime \times K_T^\prime$ is in the
   relaxed relation.

The classic adaptor signature use case goes like this. Alice and Bob perform all the following by secure
side-channel except the final step. First, Bob wants a signature $\xi$ from Alice's key $vk$ on a message $m$ of his
choice posted on a public bulletin board, and Alice wants Bob to give her a secret witness $\underline{w}$ in exchange
for a signature. The secret witness $\underline{w}$ is similar to a private key, and it has a corresponding public
statement $s = \langle \underline{a}, \underline{w} \rangle$, which is akin to a public key. Next, Bob sends Alice $m$,
$s$, and a proof of knowledge of a witness $\underline{w}$ corresponding to $s$, and he asks for a _pre-signature_ from
Alice, say $\hat{\xi}$. This $\hat{\xi}$ is a commitment from Alice to $s$ with properties similar to a signature.
Indeed, $\hat{\xi}$ can be "pre-verified" against the message $m$, the public statement $s$, and Alice's key $vk$.
Moreover, if anyone other than Alice and Bob learn of this $\hat{\xi}$, then it is possible that Alice could be "
scooped;" see below. After that, Alice securely sends Bob $\hat{\xi}$, and he _adapts_ the pre-signature to a signature
$\xi$ using his secret $\underline{w}$. Now, Bob can post $\xi$ to the public bulletin board whenever he chooses.
Lastly, if Bob ever posts $\xi$ to the public bulletin board, Alice can download $\xi$ and use it together with the
secret $\hat{\xi}$ to extract the secret witness $\underline{w}$.

One key point here is that two pieces of secret information is required to compute a signature: both the signing key
$sk$ and the secret witness $\underline{w}$. Furthermore, these two pieces of information are owned by two different
parties who do not necessarily trust each other.

With this use-case in mind, we present an informal definition of an adaptor signature scheme, which uses an underlying
hard relation as a sub-scheme. An adaptor signature scheme is a tuple of algorithms $(\texttt{Setup}, \texttt{Keygen},
\texttt{PreSign}, \texttt{PreVerify}, \texttt{Adapt},
\texttt{Sign}, \texttt{Verify}, \texttt{Extract})$ which informally work as follows.

0. The setup algorithm $\texttt{Setup}$ includes a hard relation $(\texttt{Setup}^*, \texttt{RelationGen},
   \texttt{RelationVerify})$ and an unforgeable signature scheme $(\texttt{Setup}^\prime, \texttt{Keygen},
   \texttt{Sign}, \texttt{Verify})$ in its output, as well as a description of a presignature set $\widehat{\Xi}$.
1. $\texttt{PreSign}((sk, vk), s, m) \to \hat{\xi} \in \widehat{\Xi}$. Input a keypair $(sk, vk) \in K_S \times K_V$, a
   public statement $s \in K_T$, and a message $m \in M$, and outputs a pre-signature, $\hat{\xi} \in \widehat{\Xi}$.
2. $\texttt{PreVerify}(vk, s, m, \hat{\xi}) \to b \in \left\{0, 1\right\}$. Input keypair $(sk, vk)$, a public statement
   $s$, a message $m$, and a pre-signature $\hat{\xi}$, and outputs a bit indicating the validity of the pre-signature.
3. $\texttt{Adapt}(\hat{\xi}, \underline{w}) \to \xi \in \Xi$ inputs a pre-signature $\hat{\xi}$ and a secret witness
   $\underline{w}$ and outputs a signature $\xi \in \Xi$.
4. $\texttt{Extract}(\xi, \hat{\xi}) \to \underline{w} \in K_W$. Input a pre-signature $\hat{\xi}$, and a signature
   $\xi$, and outputs a secret witness $\underline{w}$.

Alternatively, $\texttt{Extract}$ may produce a relaxed witness $\underline{w}^\prime \in K_W^\prime$.

### Cross-Chain Atomic Swaps

Alice and Bob can compute cross-chain atomic swaps using the following approach.

1. Alice selects a secret witness $\underline{w}$ with corresponding public statement $s$, and sends $s$ to Bob together
   with a proof of knowledge that she knows a secret witness for $s$; if Bob is unconvinced, he proceeds no further.
2. Alice posts a time-locked transaction $T_A$ on the first chain with a long time-lock that is claimable by Bob if he
   reveals $\underline{w}$.
3. After Bob sees Alice's transaction posted on-chain, Bob posts a transaction $T_B$ on the second chain with a shorter
   time-lock that is claimable by Alice if she reveals $\underline{w}$.
4. Alice computes a pre-signature $\hat{\xi}_A$ signing $T_A$ and Bob computes a pre-signature $\hat{\xi}_B$ signing
   $T_B$ and send them to each other via secure side channel.
6. Alice adapts $\hat{\xi}_B$ with $\underline{w}$ to obtain a signature $\xi_B$. Alice can post this on the second
   blockchain to obtain her funds.
7. After seeing $\xi_B$ posted on the second blockchain, Bob can download $\xi_B$ and use $\hat{\xi}_B$ to compute
   $\underline{w} = \texttt{Extract}(\xi_B, \hat{\xi}_B)$. Now, Bob can adapt $\hat{\xi}_A$ with $\underline{w}$ to
   obtain $\xi_A$, which he can post on the first blockchain to claim his funds.

Note that since the time-lock on Bob's transaction is short, Alice has time to claim her funds before the time-lock
elapses. Moreover, after careful thought, it should be clear that either both parties get what they want, or neither
party does. By building refund functionality into the transaction protocol, the users can also guarantee they get their
funds back if something goes wrong. This protocol is summarized in the following diagram.

![A graphical depiction of cross-chain swaps using adaptor sigantures.](ccswap.png)

For example, $\underline{w}$ may be a secret witness that plays a dual role, where it is both a bitwise description of a
small-norm vector in a Module-SIS setting, and also a pre-image for a hash in a pay-to-hash-pre-image cryptocurrency.

### Payment Channel Networks

We can employ the anonymous multi-hop lock (AMHL) approach to developing a payment channel network with lattice-based
approaches.

1. A user selects random secret witnesses $\underline{w}_0, \underline{w}_1, \underline{w}_2, ...$ for each intermediary
   node through which their payment channel will route.
2. The user computes the sequence of partial sums $\sigma_0 = \underline{w}_0, \sigma_1 = \sigma_0 + \underline{w}_1,
   \sigma_2 = \sigma_1 + \underline{w}_2, ...$ and computes their dot product against the key challenge, $P_0 = a *
   \sigma_0, P_1 = a *
   \sigma_1, ...$.
3. The user sends $(P_0, P_1, \underline{w}_1)$ to the first intermediary node, $(P_1, P_2, \underline{w}_2)$ to the
   second intermediary node, $(P_2, P_3, \underline{w}_3)$ to the third intermediary node, and so on.
4. The user sends $P_N$ to the final recipient.
5. Each intermediary verifies that $a * \underline{w}_i + P_{i-1} = a * \sigma_i = P_i$.
6. To make a payment to the final recipient, the user makes a conditional payment to the first intermediary node, who
   then makes a conditional payment to the second intermediary node, and so on. Once all these are placed, the user
   reveals $\underline{w}_N$ to the final recipient. The final recipient adapts a pre-signature to a signature and posts
   it. This allows the last intermediary node to extract a secret witness, from which they can adapt their pre-signature
   to a signature and post on the blockchain. This proceeds backward iteratively until all conditional payments are
   completed.

Just like a positive cloud-to-ground lightning strike actually sees electrons moving from the ground to the sky, the
revelation of secret witnesses proceeds backward from the final recipient to the sending user like a lightning strike.

### Security Models

Just as before, without additional properties, the signature scheme is not useful. We take the tactic where we attempt
to provide the reader intuition about how adaptor signatures are used in order to inform their security properties.

In both of the protocols we have described here, users require that pre-signatures can be adapted to valid signatures,
and users require that signature-pre-signature pairs are enough to extract witnesses. And, of course, we require that
signatures are unforgeable. Of course, if signatures are unforgeable and they are easy to adapt from pre-signatures,
then unforgeability of signatures will also imply unforgeability of pre-signatures. This is why protocols with adaptor
signatures can (sometimes) be thought of has having built-in authentication by default. Thus, in addition to
unforgeability and correctness, we have the notions of _pre-signature adaptability_ and _(weak) witness extractability_.

#### Correctness

Correctness merely says that the sub-schemes (the hard relation and the signature scheme) are both correct and if honest
parties use the adaptor signature scheme as expected, everything works out. In particular:

1. A pre-signature $\hat{\xi}$ computed from $\texttt{PreSign}$ with an honest keypair $(sk, vk) \leftarrow
   \texttt{Keygen}$ and a statement $s$ on message $m$ passes pre-verification: $\texttt{PreVerify}((sk, vk), s, m,
   \texttt{PreSign}((sk, vk), s, m)) = 1$.
2. Adapting an honestly computed pre-signature $\hat{\xi}$ with an honest witness-statement pair $(w, s)$ results in a
   valid signature $\xi$: $\texttt{Verify}(vk, s, m, Adapt(\hat{\xi}, \underline{w})) = 1$.
4. Extracting a witness from this pre-signature and adapted pre-signature results in a valid (although possibly relaxed)
   witness: $\texttt{RelationVerify}(\texttt{Extract}(\hat{\xi}, \xi), s) = 1$.

Like in our previous articles, we learn about a security property P by _negating_ P and interpreting what _that_ means.
In this spirit, if the scheme is not correct, then at least one of these three properties fail. If the first property
fails, then an honestly computed pre-signature on an honestly-sampled key with an honestly sampled statement from the
hard relation may fail pre-verification by an honest verifier. If the second property fails, then an honest adaption of
an honestly computed pre-signature with an honestly sampled witness from the hard relation may fail verification by an
honest verifier. If the third property fails then honestly extracting a witness from an honestly computed
signature-pre-signature pair may fail to produce a valid witness.

So, if a scheme is correct, then honest users can rest assured that their pre-signatures will pass honest
pre-verification, and they can be honestly adapted to signatures that pass honest verification, and that valid witnesses
can be extracted from such honest signature-pre-signature pairs. The remainder of the security properties ensure that we
can relax these notions to not-necessarily-honest constructions.

We note that since this property does not require any oracle access, or multiple keys or signatures, then it applies
just fine to the one-time signature environment (just so long as we make sure to emphasize that all keys and witnesses
are one-time).

#### Pre-signature adaptability

We want to ensure that it is not possible for an adversary to find a valid-looking pre-signature $\hat{\xi}$ and
statement $s$ such that the adaptation with a valid witness $\underline{w}$ for $s$
$\xi \leftarrow \texttt{Adapt}(\hat{\xi}, \underline{w})$ fails signature verification. This prevents an adversary from
convincing a victim who knows a valid witness $\underline{w}$ for $s$ to accept a valid-looking $\hat{\xi}$ from which
the victim cannot produce a valid signature. Pre-signature adaptability is the property that, for a statement $s$ and
a valid pre-signature $\hat{\xi}$ on $s$, users with a valid witness $\underline{w}$ for $s$ can adapt $\hat{\xi}$ to
obtain a valid signature $\xi$, even if $\hat{\xi}$ was not honestly computed. We call this property pre-signature
adaptability.

In symbols, if $vk$ is any verification key, $s$ is any statement, $\underline{w}$ is any witness, $m$ is any message,
$\hat{\xi}$ is any pre-signature, $A$ is the event that $\texttt{PreVerify}(vk, s, m, \hat{\xi}) = 1$, $B$ is the event
that $\texttt{RelationVerify}(\underline{w}, s) = 1$, and $C$ is the event that
$\texttt{Verify}(vk, s, m, \texttt{Adapt}(\hat{\xi}, \underline{w})) = 1$, we want the probability
$\mathbb{P}\left[C \mid A \cap B\right] = 1$ (or at least negligibly close to 1).

We note that since this property does not require any oracle access, or multiple keys or signatures, then it applies
just fine to the one-time signature environment (just so long as we make sure to emphasize that all keys and witnesses
are one-time).

#### (Weak) Witness Extractability

We want to ensure that it is not possible for an adversary to publish a valid pre-signature $\hat{\xi}$ and a valid
signature $\xi$ on the same data (including the same statement $s$) without giving away a valid (although perhaps
relaxed) witness $\underline{w}^\prime$ to $s$. Witness extractability (or weak witness extractability, respectively) is
the property that, for a statement $s$ and a valid signature-pre-signature pair $(\xi$ and $\hat{\xi})$ on $s$, the
(weak) witness extracted with $\underline{w}^\prime \leftarrow \texttt{Extract}(\hat{\xi}, \xi)$ is valid, even if the
signatures were not computed honestly.

In symbols, if $vk$ is any verification key, $m$ is any message, $s$ is any statement, $\hat{\xi}$ is any pre-signature,
$\xi$ is any signature, $A$ is the event that $\texttt{PreVerify}(vk, s, m, \hat{\xi}) = 1$, $B$ is the event that
$\texttt{Verify}(vk, m, \xi) = 1$, $C$ is the event that $\underline{w} \leftarrow \texttt{Extract}(\xi,
\hat{\xi})$, $D$ is the event that $\texttt{RelationVerify}(\underline{w}, s) = 1$, and $D^\prime$ is the event that
$\texttt{RelationVerify}^\prime(\underline{w}, s) = 1$, then witness extractability is when
$\mathbb{P}\left[D \mid A \cap B \cap C] = 1$ (or is negligibly close to 1) and weak witness extractability is when
$\mathbb{P}\left[D^\prime \mid A \cap B \cap C] = 1$ (or is negligibly close to 1).

In the schemes we discuss here, we actually do not attain the property of witness extractability, we only have
_weak witness extractability_, where a _relaxed_ witness can be extracted. In the above model of the lightning network,
using weak witness extractability in a lattice setting, extracted witnesses grow in size until they are too large to be
useful as secret witnesses. This places a natural maximum length on the chain of intermediary payment nodes.

We note that since this property does not require any oracle access, or multiple keys or signatures, then it applies
just fine to the one-time signature environment (just so long as we make sure to emphasize that all keys and witnesses
are one-time).

#### Unforgeability

It is necessary that the underlying signature scheme is unforgeable in order for the adaptor scheme to be unforgeable.
Indeed, if the underlying signature scheme is unforgeable, then even an algorithm with access to $\texttt{PreSign}$ and
$\texttt{Adapt}$ won't be able to construct a valid signature on a challenge key. However, this is not sufficient.

Indeed, in unforgeability definitions, we grant the adversary the benefit of the doubt by allowing this adversary to
persuade honest users to publish signatures on messages chosen by the attacker. This is modeled by giving the adversary
signature oracle access. In the adaptor signature scheme, the adversary should also be granted pre-signature oracle
access so that the model of unforgeability grants the adversary a similar benefit. We grant the adversary even further
benefits, by giving the adversary one free pre-signature oracle query. This models the situation that the adversary is
only concerned with forgeries that appears to be an adaptation a specific oracle-generated pre-signature. In this case,
the forger is granted one free pre-signing oracle query, is otherwise granted unlimited and adaptive access to both the
signing and the pre-signing oracle, and is asked to produce any signature on any challenge key on any message that was
not queried to either the signing or the pre-signing oracles.

However, in the one-time signature case, oracle access is limited to one-time access. In the adaptor signature case,
we think this means that both oracles should only be queriable once, including the free pre-signature oracle query 
granted to the adversary. However, one-time extensions of adaptor signatures is a novel and new area of applied crypto-
graphy, so we can't say much more until further research rolls in.

### Implementing

We note that we can rather easily adapt the previously described one-time lattice-based signature scheme to construct
a protocol that is _nearly_ a one-time adaptor signature scheme, but not quite. In fact, the scheme consists of a tuple
of the following algorithms.

0. The setup algorithm includes the output from the setup algorithm from our first article on one-time signature
   schemes, but with some additional data. In particular, we describe a pair of nested hard relations.
1. $\texttt{Sign}((sk, vk), (\underline{w}, s), m) \to \xi \in \Xi$ modifies the LMSigs signature algorithm by pre-
   pending $s$ to $m$ before computing the signature challenge, and summing the witness $\underline{w}$ with the result.
   That is to say, the signature is now $\xi = \underline{x} \cdot c + \underline{y} + \underline{w}$ where we compute
   $c = F(vk, s, m)$ instead of $c = F(vk, m)$.
2. $\texttt{Verify}(vk, s, m, \xi) \to \left\{0, 1\right\}$ modifies the LMSigs verification algorithm by merely pre-
   pending $s$ to $m$ before verifying, and by modifying the verification equation to account for how $\texttt{Sign}$
   works. That is to say, we check the bounds on $\xi$ and check if $\langle \underline{a}, \xi \rangle = X c + Y + s$,
   where $c = F(vk, s, m)$.
3. $\texttt{PreSign}((sk, vk), s, m) \to \hat{\xi} \in \widehat{\Xi}$ computes the signature challenge $c$ as in
   $\texttt{Sign}$ and ouptuts $\hat{\xi} = \underline{x} c + \underline{y}$.
4. $\texttt{PreVerify}(vk, s, m, \hat{\xi}) \to \left\{0, 1\right\}$ checks the bound on $\hat{\xi}$ and
   checks that $\langle \underline{a}, \hat{\xi}\rangle = Xc + Y$ where $c$ is computed as in $\texttt{Sign}$ and
   $\texttt{PreSign}$.
5. $\texttt{Adapt}(\hat{\xi}, \underline{w}) \to \xi = \hat{\xi} + \underline{w}$ outputs the sum.
6. $\texttt{Extract}(\hat{\xi}, \xi) \to \underline{w}^\prime = \xi - \hat{\xi}$ outputs the difference.

This is formally not an adaptor signature scheme; although it employs algorithms that _wrap_ the LMSigs signature scheme
it does not use those algorithms directly as a proper sub-scheme. Users have access to all the same algorithms as in an 
adaptor signature scheme. We think that the different security properties of adaptor signature schemes still carry 
through with certain technical modifications to the definitions. 

So it seems to us that this approach can lead to a quantum-resistant one-time adaptor signature scheme... provided, of 
course, that a suitable one-time extension to adaptor signature security definitions can be found and agreed upon. We
emphasize that one-time extensions of adaptor signatures is a novel and new area of applied cryptography and that we 
can't say much more until further research rolls in.

### Conclusion

With adaptor signature schemes, a lot of intra- and inter-chain commerce is enabled based on the claim-or-refund
transaction model (see [[1]](https://link.springer.com/chapter/10.1007/978-3-662-44381-1_24)). MITCHELL: HELP HERE
PLEASE

### References

[[1]](https://link.springer.com/chapter/10.1007/978-3-662-44381-1_24) Bentov, Iddo, and Ranjit Kumaresan. "How to use
bitcoin to design fair protocols." Annual Cryptology Conference. Springer, Berlin, Heidelberg, 2014.

