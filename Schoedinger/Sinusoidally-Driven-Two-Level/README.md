For a two-level system driven by a monochromatic field
$$
\begin{equation}
    v=\frac{\epsilon_0}{2}\{\exp(i\omega_0 t)+\exp(-i\omega_0 t)\},\epsilon_0=\mu \mathcal E_0,
\end{equation}
$$
the Hamiltonian and rotating wave approximation (RWA) read
$$
\begin{equation}\label{eq:H}
    H=
    \begin{bmatrix}
        E_1&v(t) \\
        v(t)&E_2
    \end{bmatrix}\approx
    \begin{bmatrix}
        E_1&\frac{\epsilon_0}{2}e^{i\omega_0 t} \\
        \frac{\epsilon_0}{2}e^{-i\omega_0 t}&E_2
    \end{bmatrix}.
\end{equation}
$$
The time-evolution Schrödinger reads,
$$
i\hbar\frac{\partial}{\partial t}\vec \psi=\hat H \vec \psi.
$$
Under RWA it can be exactly solved by ansatz
$$
\begin{equation}\label{eq:3.ansatz}
    \vec \psi_i
    =
    \begin{bmatrix}
    a^ie^{i\omega_0t}\\
    b^i
    \end{bmatrix}, i=1,2
\end{equation}
$$
