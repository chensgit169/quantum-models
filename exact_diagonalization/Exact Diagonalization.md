# Exact Diagonalization

Chen Wei, 28 Dec. 2023

As we learned in quantum mechanics, operators are represented as matrices under a given set of basis. Moreover, static Schroedinger equations boil down to finding eigenstates and eigenvalues of the Hamiltonian. Method of Exact Diagonalization (ED) is nothing but numerically solving eigen-problem of matrices. To implement ED, and even more generally, to tackle quantum problems through numerics, the very first question is how to represent the Hilbert space and the operators acting on it. Here we would take the Hubbard model in the following form as example.
$$
H_{\text{HD}}=-t\sum_{i\sigma\delta}c_{i\sigma}^{\dagger}c_{i+\delta,\sigma}+U\sum_in_{i\uparrow}n_{i\downarrow}
$$
The $\delta$​ connects sites where electron hopping may go. A most natural choice of basis which span the total Hilbert space is 
$$
\mathcal{H}=<|\Psi_{I_{\uparrow}I_{\downarrow}}\rangle
=\prod_{i\in  I_{\uparrow}}c_{i{\uparrow}}^\dagger
\prod_{i\in  I_{\downarrow}}c_{i{\downarrow}}^\dagger|0\rangle>
$$
where $I_{\uparrow}$ and $I_{\downarrow}$ are sets of occupies sites by spin-up and spin-down electron, respectively. We may encode  them as binary strings with length $2N$, where $N$ is the total site number. 

Notice that $H_{\text{HD}}$ can be written as the summation of "local Hamiltonian" on each sites, namely
$$
H_{\text{HD}}=\sum_ih_i\\
\text{where}\  \  h_i=-t\sum_{\sigma\delta}c_{i\sigma}^{\dagger}c_{i+\delta,\sigma}+Un_{i\uparrow}n_{i\downarrow}
$$
And $\mathcal{H}$ is actually a product of "local Hilbert spaces",
$$
\mathcal{H}=\otimes_i\mathcal{H_i}\\
\text{where}\  \ \mathcal{H_i}
=\{
|0_i\rangle, c_{i{\uparrow}}^\dagger|0_i\rangle, 
c_{i\downarrow}^\dagger|0\rangle,
c_{i\uparrow}^\dagger c_{i\downarrow}^\dagger|0\rangle  \}
$$
from which the dimension of $\mathcal{H}$ can be immediately seen to be $4^N$. It also suggests that it is useful to view a general state $|\psi\rangle$ as a order-$N$ tensor, more precisely, 
$$
|\psi\rangle=\sum_{a_1, a_2,...,a_N}\psi_{a_1, a_2,...,a_N}|a_1, a_2,...,a_N\rangle
$$
where $a_i=i_{\uparrow}i_{\downarrow}$ is the quaternary encoding of basis. And $\psi_{a_1, a_2,...,a_N}$ is exactly what we numerically store in ED. 

It's clear that $h_i$ acts actually on  $\mathcal{H_i}$ and $\{\mathcal{H_{i+\sigma}}\}$, a sub-Hilbert space with rather limited dimension, while for other local spaces the role of $h_i$ is the identity. In the language of tensor, only indices $i$ and $\{i+\sigma\}$ of $\psi_{a_1, a_2,...,a_N}$ are relevant when applying $h_i$. This property is very commonly seen in quantum models defined on lattices and brings us much convenience, especially when considering only the short-range interactions. 

To this point the readers should have gained a feeling about how to numerically encode states and operators. In the rest of this note, we will elaborate more about the key points of implementing ED.

<div style="page-break-after:always;"></div>

A critical process we omitted above is how to solve the eigen-problem of a given matrix in numerics. However, here we would rather view it as the issue of applied mathematicians. What we need to do is only to notice some key properties of our "matrix" and to invoke ready-made computational packages.  

If we explicitly write down the matrix representation of $H$, it must be a sparse one. However, we don't have to store the matrix, whose size grows exponentially with $N$. For this aim, we found ``python`` could be easily accessible and pedagogical
