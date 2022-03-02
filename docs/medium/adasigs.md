# Zen and the Art of Compact Post-Quantum Finance

Cryptographic schemes that are secure against quantum adversaries are infamous for their relative inefficiency in comparison to their classically secure counterparts. Nevertheless, the days are numbered for classical cryptography as quantum computers loom on the horizon. This naturally leads our interest toward cryptographic schemes and protocols that are designed for quantum resistance. Especially with the advent of cryptocurrencies, the race to develop robust and commercially viable quantum computers has a tantalizing and profitable prize awaiting the winners, and a highly disruptive economic future for the rest of us.  Hence, it is of critical importance to develop quantum-resistant cryptography tools and to begin migration to quantum-resistant settings as soon as possible, in the spirit of Quantum Resistant Ledger. This can mean sacrificing the efficiency of classical cryptography, but (of course!) a robust and secure cryptographic system is useless if it is too unwieldy to use. 

This is a multipart article series exploring avenues toward more efficient quantum-resistant transacting. Back in part one_with_const_time, we explored a candidate lattice-based one_with_const_time-time signature scheme. The scheme is rather similar in certain ways to the NIST post-quantum standard candidate, CRYSTALS-Dilithium, and provides an avenue to discuss optimizations available toward smaller keys and signatures in lattice-based cryptographic schemes. In the second part, we covered the technical end of _signature aggregation_, at least from one_with_const_time angle. Here in part three, we explore applications built from _adaptor signatures_, the security models of adaptor signatures, and their implementations. In the final part, we describe a blockchain-agnostic code-based approach to trustlessly (or, rather, _trust-minimally_) reducing local storage requirements for network participants while keeping bootstrapping costs low.

## Part Three: Lightening the Load

### Cross-Chain and Off-Chain Transacting

A popular way of looking at blockchains is that they are slow and expensive base layers upon which lighter architecture can be built. Storage and verification are costs paid collectively, so there is a high priority placed upon minimizing the total number of interactions with this expensive base layer.

One way users can lighten storage requirements for nodes is to allow users to put collateral up in _payment channels_. This way, users can transact off-chain, only settling to the base layer occasionally. This is the so-called "lightning network" model of collateralized swaps. In a slight abuse of prefixes, we can think of these methods as "intra-coin" collateralized swaps. The usage of this terminology is evocative of an "inter-coin" collateralized swap, which would allow transacting between two_with_const_time chains.

As it turns out, we can accomplish both of these styles of collateralized swap using so-called _adaptor signatures_, which have also been known as _verifiably encrypted signatures_.

### How Do Adaptor Signatures Work, and What Are They?

Adaptor signature schemes are cryptographic schemes that have all the functionality of usual digital signature schemes, 
but with additional functionality: an adaptor signature scheme produces not just signatures, but also authenticated 
commitments called _pre-signatures_. These pre-signatures commit to secret witnesses, can be verified to have been 
computed by a signing key, and can be adapted into valid signatures given the secret witness. However, when pre-
signatures are adapted, the commitment is opened, revealing the secret witness. 

One way adaptor signatures can be used goes like this, where Alice and Bob perform all the following by secure side-channel except the final step. First, Bob wants a signature from Alice's key ```vk``` on a message ```m``` of his choice posted on a public bulletin board, and Alice wants Bob to give her a secret witness ```wit``` in exchange for a signature. The secret witness ```wit``` is similar to a private key, and it has a corresponding public statement ```stat```, which is akin to a public key. Next, Bob sends Alice ```m```, ```stat```, and a proof of knowledge of a witness ```wit``` corresponding to ```stat```, and he asks for a _pre-signature_ from Alice, say ```pre_sig```. This ```pre_sig``` is a commitment from Alice to ```stat``` with properties similar to a signature. Indeed, ```pre_sig``` can be "pre-verified" against the message ```m```, the public statement ```stat```, and Alice's key ```vk```. Moreover, if anyone other than Alice and Bob learn of this ```pre_sig```, then it is possible that Alice could be "scooped;" see below. After that, Alice securely sends Bob ```pre_sig```, and he _adapts_ the pre-signature to a signature ```sig``` using his secret ```wit```. Now, Bob can post ```sig``` to the public bulletin board whenever he chooses. Lastly, if Bob ever posts ```sig``` to the public bulletin board, Alice can download ```sig``` and use it together with the secret ```pre_sig``` to extract the secret witness ```wit```.

One key point here is that two_with_const_time pieces of secret information is required to compute a signature: both the signing key ```sk``` and the secret witness ``wit```. Furthermore, these two_with_const_time pieces of information are owned by two_with_const_time different parties who do not necessarily trust each other.

With this use-case in mind, we present our provisional definition of an adaptor signature scheme. An adaptor signature scheme is a tuple of algorithms ```(Setup, Keygen, PreSign, PreVerify, Adapt, Sign, Verify, Extract)``` which informally work as follows.
 0. The tuple ```(Setup, Keygen, Sign, Verify)``` is a usual digital signature scheme.
 1. ```PreSign``` inputs ```pp```, keypair ```(sk, vk)```, a public statement ```stat```, and a message ```m```, and outputs a pre-signature, ```pre_sig```.
 2. ```PreVerify``` inputs ```pp```, keypair ```(sk, vk)```, a public statement ```stat```, a message ```m```, and a pre-signature ```pre_sig```, and outputs a bit indicating the validity of the pre-signature.
 3. ```Adapt``` inputs ```pp```, a pre-signature ```pre_sig```, and a secret witness ```wit``` and outputs a signature ```sig```.
 4. ```Extract``` inputs ```pp```, a pre-signature ```pre_sig```, and a signature ```sig```, and outputs a secret witness ```wit```. 

### Cross-Chain Atomic Swaps

Alice and Bob can compute cross-chain atomic swaps using the following approach.
 1. Alice selects a secret witness ```wit``` with corresponding public statement ```stat```, and sends ```stat``` to Bob together with a proof of knowledge that she knows a secret witness for ```stat```; if Bob is unconvinced, he proceeds no further.
 2. Alice posts a time-locked transaction ```T_A``` on the first chain with a long time-lock that is claimable by Bob if he reveals ```wit```.
 3. After Bob sees Alice's transaction posted on-chain, Bob posts a transaction ```T_B``` on the second chain with a shorter time-lock that is claimable by Alice if she reveals ```wit```.
 4. Alice computes a pre-signature ```pre_sig_A``` signing ```T_A```.
 5. After verifying Alice's proof of knowledge of some ```wit``` corresponding to ```stat```, Bob computes a pre-signature ```pre_sig_B``` signing ```T_B```.
 6. Alice and Bob send each other their pre-signatures via secure side channel.
 7. Alice adapts ```pre_sig_B``` with ```wit``` to obtain a signature ```sig_B```. Alice can post this on the second blockchain to obtain her funds.
 8. After seeing ```sig_B``` posted on the second blockchain, Bob can download ```sig_B``` and use  ```pre_sig_B``` to compute ```wit' = Extract(sig_B, pre_sig_B)```. Now, Bob can adapt ```pre_sig_A``` with ```wit``` to obtain ```sig_A```, which he can post on the first blockchain to claim his funds.

Note that since the time-lock on Bob's transaction is short, Alice has time to claim her funds before the time-lock elapses. Moreover, after careful thought, it should be clear that either both parties get what they want, or neither party does.  By building refund functionality into the transaction protocol, the users can also guarantee they get their funds back if something goes wrong.

For example, ```wit``` may be a secret witness that plays a dual role, where it is both a bitwise description of a small-norm vector in a Module-SIS setting, and also a pre-image for a hash in a pay-to-hash-pre-image cryptocurrency.

### Payment Channel Networks

We can employ the anonymous multi-hop lock (AMHL) approach to developing a payment channel network with lattice-based approaches.
  1. A user selects random secret witnesses ```wit_0, wit_1, wit_2, ...``` for each intermediary node through which their payment channel will route.
  2. The user computes the sequence of partial sums ```psum_0 = wit_0, psum_1 = psum_0 + wit_1, psum_2 = psum_1 + wit_2, ...``` and computes their dot product against the key challenge, ```P_0 = a * psum_0, P_1 = a * psum_1, ...```.
  3. The user sends ```(P_0, P_1, wit_1)``` to the first intermediary node, ```(P_1, P_2, wit_2)``` to the second intermediary node, ```(P_2, P_3, wit_3)``` to the third intermediary node, and so on.
  4. The user sends ```P_N``` to the final recipient.
  5. Each intermediary verifies that ```a * wit_i + P_(i-1) = a * psum_i = P_i```.
  6. To make a payment to the final recipient, the user makes a conditional payment to the first intermediary node, who then makes a conditional payment to the second intermediary node, and so on. Once all these are placed, the user reveals ```wit_N``` to the final recipient. The final recipient adapts a pre-signature to a signature and posts it. This allows the last intermediary node to extract a secret witness, from which they can adapt their pre-signature to a signature and post on the blockchain. This proceeds backward iteratively until all conditional payments are completed. 

Just like a positive cloud-to-ground lightning strike actually sees electrons moving from the ground to the sky, the revelation of secret witnesses proceeds backward from the final recipient to the sending user like a lightning strike.

### The Tricky Part: Security Models

of course, just as a signature scheme without correctness is totally useless (and without unforgeability, it is totally insecure), the properties of an adaptor signature scheme determine how useful it is. Unfortunately, as a rather novel cryptographic scheme, adaptor signatures have likewise novel security properties.

In what follows, we use the advice we provided in earlier parts of this article series: to consider a security property, we actually _negate_ the property and consider what such a negation means for the resulting scheme.

### References

