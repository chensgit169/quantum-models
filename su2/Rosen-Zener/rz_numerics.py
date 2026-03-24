from su2.common.su2_integrator import u
from magnus_1st import AdiabaticSolver

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


"""
Rosen-Zener model solved in Schrödinger picture

H = 1/2 * [[d, f(t)],
           [f(t), d]]
where f(t) = g * sech(t/T), T=1 chosen as time unit. 

Last updated: 2026-Mar-14
"""


# initial state is the eigen state of a*t
psi_0 = np.array([0, 1])


def h_landau_zener(t, g, d) -> np.ndarray:
    # Landau-Zener Hamiltonian
    return np.array([g / np.cosh(t), 0, d])


def p_final_exact(g, d):
    """Exact transition probability for Rosen-Zener model"""
    alpha = d / 2
    beta = g / 2
    return np.sin(np.pi * beta) ** 2 / np.cosh(np.pi * alpha) ** 2


def p_final_numerical(g, d, t_lim=20, Nt=1000):
    """Numerical transition probability for Rosen-Zener model"""
    ut = u(t_lim, -t_lim, Nt, h_landau_zener, g, d)
    amp_final = psi_0.conjugate() @ (ut @ psi_0)
    return 1 - np.abs(amp_final) ** 2


def p_magnus_1st(g, d, t_lim=20, Nt=1000):
    solver = AdiabaticSolver(g, d)
    t_vals = np.linspace(-t_lim, t_lim, Nt)
    beta_magnus = solver.wkb_psi_tp(t_vals)['beta']
    return np.abs(beta_magnus[-1]) ** 2


def demo_pulse_area():
    """
    demonstrate the dependence of transition probability on pulse area g,
    """
    d = 0.1

    g_vals = np.linspace(0, 10, 200)

    p_exact = p_final_exact(g_vals, d)
    p_numerical = np.array([p_final_numerical(g, d) for g in tqdm(g_vals)])
    p_wkb = np.array([p_magnus_1st(g, d) for g in tqdm(g_vals)])

    plt.figure(figsize=(6, 4.5))
    plt.plot(g_vals, p_exact, label='Exact')
    plt.plot(g_vals, p_numerical, 'o', label='Numerical', color='red')
    # plt.plot(g_vals, p_wkb, 'x-', label='Magnus 1st')

    plt.xlabel(r'$g$')
    plt.ylabel(r'$P$')
    plt.title(f'Transition Probability vs g, d={d}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def demo_detuning():
    """
    demonstrate the dependence of transition probability on detuning d,
    """
    g = 1

    d_vals = np.linspace(-3, 3, 100)

    p_exact = np.array([p_final_exact(g, d) for d in d_vals])
    p_numerical = np.array([p_final_numerical(g, d) for d in tqdm(d_vals)])

    plt.figure(figsize=(6, 4.5))
    plt.plot(d_vals, p_exact, label='Exact')
    plt.plot(d_vals, p_numerical, 'o', label='Numerical', color='red')
    plt.xlabel(r'$d$')
    plt.ylabel(r'$P$')
    plt.title(f'Transition Probability vs d, g={g}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    demo_pulse_area()
    # demo_detuning()
