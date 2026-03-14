import numpy as np

from su2.common.su2_integrator import u
from exact_solution import quasi_energy

"""
Solve semiclassical Rabi model numerically.

Last modified: 2026-Mar-11
"""


def h_rabi(t, d, g, omega=1):
    return np.array([g * np.cos(omega * t), 0.0, d]) / 2


def quasienergy_num(d, g, N=1000):
    """Compute quasienergies via Floquet operator over one driving period T=2π/ω."""
    T = 2*np.pi
    U_T = u(T, 0.0, N, h_rabi, d, g)
    # eigenvalues of Floquet operator
    vals, _ = np.linalg.eig(U_T)
    # quasienergy: -i log(lambda)/T
    qe = -np.angle(vals)/T
    # sort for consistency
    return np.sort(qe)


def eps_plot(d, N):

    import matplotlib.pyplot as plt

    g_vals = np.linspace(2.0913633942, 2.0913633943, 50)

    eps_all = []

    for g in g_vals:
        qe = quasienergy_num(d, g, N)
        eps_all.append(qe)

    eps_all = np.array(eps_all)  # shape (len(g_list), 2)

    plt.figure(figsize=(6, 4))

    eps_exact = np.array([quasi_energy(g, d) for g in g_vals])

    # plt.plot(g_vals, d / 2 * j0(g_vals),
    #          '--', label=r'$\frac{\Delta}{2}J_0(\frac{g}{\omega})$')

    # Plot
    plt.plot(g_vals, eps_exact, label='Exact')
    plt.plot(g_vals, eps_all[:, 0], label=r'$\epsilon_1$')
    plt.plot(g_vals, eps_all[:, 1], label=r'$\epsilon_2$')
    plt.xlabel(r'$g$')
    plt.ylabel('Quasienergy')
    plt.title(f'Quasienergy vs g, d={d}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    eps_plot(d=1, N=1000)
