# Two-level system in a monochromatic field

[TOC]

## Theory

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
        -\frac{E}{2}&v(t) \\
        v(t)&\frac{E}{2}
    \end{bmatrix}\approx
    \begin{bmatrix}
        \frac{E}{2}&\frac{\epsilon_0}{2}e^{i\omega_0 t} \\
        \frac{\epsilon_0}{2}e^{-i\omega_0 t}&-\frac{E}{2}
    \end{bmatrix}.
\end{equation}
$$
The time-evolution Schrödinger reads,
$$
i\hbar\frac{\partial}{\partial t}\vec \psi=\hat H \vec \psi.
$$
Under RWA it can be analytically solved, the the independent special solutions are:
$$
\begin{equation}
    \vec \psi^\pm(t)
    =
    \begin{bmatrix}
    e^{+i\frac{\omega_0}{2}t} & 0\\
    0 & e^{-i\frac{\omega_0}{2}t}
    \end{bmatrix}e^{\mp i\frac{\Omega_0t}{2}}|\pm\rangle
\end{equation}
$$
where
$$
|+\rangle=\begin{bmatrix}
    a^+\\
    -a^-
    \end{bmatrix} \text{,  }
|-\rangle=\begin{bmatrix}
    a^-\\
    a^+
    \end{bmatrix}
$$
and
$$
a^\pm=\frac{\epsilon_0}{\sqrt{\epsilon_0^2+(\Delta \pm \Omega)^2}}\\
\Delta=E-\omega_0,\Omega=\sqrt{\Delta^2+\epsilon_0^2})^2
$$
By setting initial condition
$$
\vec \psi(0)
    =
    \begin{bmatrix}
    1\\
    0
    \end{bmatrix},
$$
we have
$$
\vec \psi(t)
    = a^+ \vec \psi^+(t)+a^- \vec \psi^+(t)
$$

## Numerics

```python
solu_rwa = solve_ivp(h_rwa, t_span, psi0, t_eval=t_eval, method='RK45')
solu_full = solve_ivp(h_full, t_span, psi0, t_eval=t_eval, method='RK45')
```

![near_resonance_strong_field](C:\Users\weich\PycharmProjects\quantum-models\Schoedinger\Sinusoidally-Driven-Two-Level\near_resonance_strong_field.png)

![near_resonance_weak_field](C:\Users\weich\PycharmProjects\quantum-models\Schoedinger\Sinusoidally-Driven-Two-Level\near_resonance_weak_field.png)

![off_resonance_strong_field](C:\Users\weich\PycharmProjects\quantum-models\Schoedinger\Sinusoidally-Driven-Two-Level\off_resonance_strong_field.png)

![off_resonance_weak_field](C:\Users\weich\PycharmProjects\quantum-models\Schoedinger\Sinusoidally-Driven-Two-Level\off_resonance_weak_field.png)
