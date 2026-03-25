import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt

from su2.magnus import magnus_su2
from su2.common import su2_exp


def _gamma(v, d):
    g = d**2 / (4 * v)
    return g


def lz_p(g):
    """ Landau-Zener formula for transition probability """
    return np.exp(-2 * np.pi * g)


def stokes_phase(g):
    """Stokes phase φ_S(g) = π/4 + g(ln g − 1) + arg Γ(1 − ig)"""
    gamma_val = gamma(1 - 1j * g)
    phi = (np.pi / 4) + g * (np.log(g) - 1) + np.angle(gamma_val)
    return np.mod(phi, 2 * np.pi)  # restrict the phase to [0, 2π)


def x_t(t, v, d):
    return v * t / d


def t_x(x, v, d):
    return x * d / v


# sinh(s) = x

def s_x(x):
    return np.arcsinh(x)


def phi_s(s, g):
    # dynamical phase
    return g * (0.5 * np.sinh(2 * s) + s)


def v_s(s, g):
    # sinh(s) = t
    return 1j * np.exp(2j * phi_s(s, g)) / (2 * np.cosh(s))


def phi_x(x, g):
    # g * (x\sqrt{1+x^2}+\ln|x+\sqrt{1+x^2}|)
    phi_val = g * (x * np.sqrt(1 + x**2) + np.log(np.abs(x + np.sqrt(1 + x**2))))
    return phi_val


def v_x(x, g):
    return 1j * np.exp(2j * phi_x(x, g)) / (2 * (1 + x ** 2))


def amplitude(a, c=0):
    alpha, beta = su2_exp(a, c)
    p = np.abs(beta) ** 2
    phase = - np.angle(alpha)
    return p, phase


def whole_axis_magnus(g_vals, N=10000, which = 's'):
    if which == 's':
        # Note that cutoff error of A1 integral is bounded by
        #       pi/2 - arctan(sinh(cutoff)) ~ 2*e^(-2*cutoff),
        # for cutoff=40, error ~ 4.25e-18.
        s_lim = 40
        a1_vals, c2_vals, a3_vals = np.array([magnus_su2(v_s, -s_lim, s_lim, g, N=N) for g in g_vals]).T
    else:
        x_lim = 500
        a1_vals, c2_vals, a3_vals = np.array([magnus_su2(v_x, -x_lim, x_lim, g, N=N) for g in g_vals]).T
    c2_vals = c2_vals.real
    return a1_vals, c2_vals, a3_vals


def demo(item='probability', recompute=False):

    g_vals = np.linspace(0.001, 5, 200)

    a1_vals, c2_vals, a3_vals = whole_axis_magnus(g_vals, N=20000, which='s')

    plt.figure(figsize=(6, 4.5))

    p_1st, _ = amplitude(a1_vals)
    p_2nd, phase_2nd = amplitude(a1_vals, c2_vals)
    p_3rd, phase_3rd = amplitude(a1_vals + a3_vals, c2_vals)

    if item == 'probability':

        p_exact = lz_p(g_vals)

        plt.plot(g_vals, p_1st, '--', label=r"FMA")
        plt.plot(g_vals, p_2nd, ':', label=r"SMA")
        plt.plot(g_vals, p_3rd, '-.', label=r"TMA")
        plt.plot(g_vals, p_exact, label=r"Exact", color='k')
        plt.ylabel(r"$P$")
        plt.xlim(0, 1)
        # plt.yscale('log')
        fig_filename = "Magnus-Landau-Zener-Probability-MA.pdf"

    elif item == 'phase':
        def phase_approx(a, c):
            theta = np.sqrt(np.abs(a) ** 2 + c ** 2)
            tan_phi = np.tan(theta) * c / theta
            phi = np.arctan(tan_phi)
            return phi

        # phase_3rd = phase_approx(a1_vals + a3_vals, c2_vals)
        phase_exact = stokes_phase(g_vals)

        # plt.text(1.2, 0.23, '(b)', fontsize=24)
        plt.plot([], [])
        # plt.plot(g_vals, phase_2nd / np.pi, ':', label=r"2nd Magnus", lw=1.2)
        # plt.plot(g_vals, phase_3rd / np.pi, '-.', label=r"3rd Magnus", lw=1.2)
        plt.plot(g_vals, phase_2nd / np.pi, ':', label=r"SMA")
        plt.plot(g_vals, phase_3rd / np.pi, '-.', label=r"TMA")
        plt.plot(g_vals, phase_exact / np.pi, label="Exact", color='k')
        # r"$\frac{\pi}{4} +\gamma(\ln\gamma -1)+\arg\left[\Gamma(1-i\delta)\right]$")
        plt.ylabel(r"$\varphi_S/{\pi}$ ")
        plt.xlim(0, 1.5)
        fig_filename = "Magnus-Landau-Zener-Phase-MA.pdf"
    elif item == 'terms':
        plt.plot(g_vals, a1_vals.imag, label=r'Im($A_1(\alpha))$')
        plt.plot(g_vals, c2_vals, label=r'$C_2(\alpha)$')
        # plt.plot(g_vals, a3_vals, label=r'$A_3(\alpha)$')
        # plt.plot(g_vals, np.pi/2-theta, label=r'$\Theta(\alpha)$')
        # plt.plot(g_vals, tan_phi, label=r'$\tan\varphi_S(\alpha)$')
        plt.title('Magnus expansion terms')
        plt.ylabel('Integral values')
        # plt.yscale('log')
        fig_filename = "Magnus-Landau-Zener-Terms.pdf"
    else:
        raise ValueError(f'Unknown item: {item}')

    plt.xlabel(r'$\gamma$')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figures/{fig_filename}', dpi=400, transparent=True)
    plt.show()


if __name__ == '__main__':
    demo('probability')
    demo('phase')
    demo('terms')
