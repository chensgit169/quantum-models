import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.special import gamma
from tqdm import tqdm

from su2.common.magnus.magnus_su2 import a3_integral, c2_integral


plt.rcParams['font.size'] = 16


"""
Magnus expansion for Landau-Zener non-adiabatic transition problem.

Last updated: Nov. 2nd, 2025
"""


data_filename = 'data/magnus_lz.npz'


############################
# Exact results
############################
def stokes_phase(alpha):
    """Stokes phase φ_S(δ) = π/4 + δ(ln δ − 1) + arg Γ(1 − iδ)， where δ = α/2"""
    delta = np.asarray(alpha) / 2
    gamma_val = gamma(1 - 1j * delta)
    phi = (np.pi / 4) + delta * (np.log(delta) - 1) + np.angle(gamma_val)
    return np.mod(phi, 2 * np.pi)  # restrict the phase to [0, 2π)


def lz_p(a):
    """ Landau-Zener formula for transition probability """
    g = a / 2
    return np.exp(-2 * np.pi * g)


############################
# Magnus expansion integrands
############################
def g_s(s):
    # auxiliary function g(s) for dynamical phase
    return 0.5 * np.sinh(2 * s) + s


def v_s(s, a):
    # sinh(s) = t
    return np.exp(1j * a * g_s(s)) / (2 * np.cosh(s))


def v_s_a1(s, a):
    # integrand for first-order Magnus term, simplified by symmetry
    return np.cos(a * g_s(s)) / np.cosh(s)


############################
# Magnus expansion integrals
# Due to symmetry, for A only need to compute imaginary parts
############################
def a1_imag(alpha, cutoff=40, N=400000):
    # Note that cutoff error is bounded by pi/2 - arctan(sinh(cutoff)) ~ 2*e^(-2*cutoff),
    # for cutoff=40, error ~ 4.25e-18.
    t = np.linspace(0, cutoff, N)
    integrand = np.cos(alpha * g_s(t)) / np.cosh(t)
    return trapezoid(integrand, t)


def c2_int(alpha, S=40, N=int(1e6)):
    return c2_integral(v_s, -S, S, alpha, N=N)


def a3_imag(alpha, S=12, N=4000):
    return a3_integral(v_s, -S, S, alpha, N=N).real


def u_mat(a1, c2=0, a3=0):
    a = 1j * (a1 + a3)
    c = c2
    theta = np.sqrt(np.abs(a) ** 2 + c ** 2)

    ratio = np.ones_like(theta)
    mask = theta >= 1e-12
    ratio[mask] = np.sin(theta[mask]) / theta[mask]

    beta = -1j * a * ratio
    alpha = np.cos(theta) - 1j * c * ratio
    return alpha, beta


def compute():
    a_vals = np.linspace(0.001, 5, 200)
    a1_vals = np.array([a1_imag(a) for a in tqdm(a_vals)])
    c2_vals = np.array([c2_int(a) for a in tqdm(a_vals)])
    a3_vals = np.array([a3_imag(a) for a in tqdm(a_vals)])

    np.savez(data_filename, alpha=a_vals, a1=a1_vals, c2=c2_vals, a3=a3_vals)
    return a_vals, a1_vals, c2_vals, a3_vals


def demo(item='probability', recompute=False):
    if not os.path.exists(data_filename) or recompute:
        compute()

    data = np.load(data_filename)
    a_vals = data['alpha']
    g_vals = a_vals / 2
    a1_vals = data['a1']
    c2_vals = data['c2']
    a3_vals = data['a3']

    plt.figure(figsize=(7, 5))

    if item == 'probability':
        p_exact = lz_p(a_vals)
        p_1st = np.sin(a1_vals) ** 2

        _, u01_2nd = u_mat(a1_vals, c2_vals)
        p_2nd = np.abs(u01_2nd) ** 2
        _, u01_3rd = u_mat(a1_vals, c2_vals, a3_vals)
        p_3rd = np.abs(u01_3rd) ** 2

        plt.plot(g_vals, p_1st, label=r"1st Magnus", lw=1.2)
        plt.plot(g_vals, p_2nd, label=r"2nd Magnus", lw=1.2)
        plt.plot(g_vals, p_3rd, label=r"3rd Magnus", lw=1.2)
        plt.plot(g_vals, p_exact, "--", label=r"$\exp(-2\pi\gamma)$", lw=1.2, color='k')
        plt.ylabel(r"$P$")
        plt.xlim(0, 1)
        # plt.yscale('log')
        fig_filename = "Magnus-Landau-Zener-Probability.pdf"

    elif item == 'phase':
        u_11_2nd, _ = u_mat(a1_vals, c2_vals)
        u_11_3rd, _ = u_mat(a1_vals, c2_vals, a3_vals)

        phase_2nd = - np.angle(u_11_2nd)
        phase_3rd = - np.angle(u_11_3rd)
        phase_exact = stokes_phase(a_vals)

        plt.plot(g_vals, phase_2nd/np.pi, label=r"2nd Magnus")
        plt.plot(g_vals, phase_3rd/np.pi, label=r"3rd Magnus")
        plt.plot(g_vals, phase_exact/np.pi, "--", label="Exact")
        # r"$\frac{\pi}{4} +\gamma(\ln\gamma -1)+\arg\left[\Gamma(1-i\delta)\right]$")
        plt.ylabel(r"$\varphi_S/{\pi}$ ")
        plt.xlim(0, 1.5)
        fig_filename = "Magnus-Landau-Zener-Phase.pdf"
    elif item == 'terms':
        plt.plot(g_vals, a1_vals, label=r'$A_1(\alpha)$')
        plt.plot(g_vals, c2_vals, label=r'$C_2(\alpha)$')
        plt.plot(g_vals, a3_vals, label=r'$A_3(\alpha)$')
        plt.title('Magnus expansion terms')
        plt.ylabel('Integral values')
        fig_filename = "Magnus-Landau-Zener-Terms.pdf"
    else:
        raise ValueError(f'Unknown item: {item}')

    plt.xlabel(r'$\gamma$')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'figures/{fig_filename}', dpi=400, transparent=True)
    plt.show()


def demo_c2():
    a_vals = np.linspace(100, 150, 100)
    c2_vals = np.array([c2_int(a) for a in tqdm(a_vals)])

    plt.figure(figsize=(7, 5))
    g_vals = a_vals / 2
    plt.plot(g_vals, c2_vals, label=r'$C_2(\alpha)$')
    plt.xlabel(r'$\alpha$')
    plt.ylabel('Integral values')
    plt.title('Second-order Magnus expansion term')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.savefig('figures/Magnus-Landau-Zener-C2.pdf', dpi=400)
    plt.show()


if __name__ == '__main__':
    demo(recompute=False, item='probability')
    # demo_c2()
