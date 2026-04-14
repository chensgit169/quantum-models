import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from su2.common import su2_exp
from su2.magnus import magnus_su2
from v_a import v_s, v_x, stokes_phase, lz_p

mpl.rcParams.update({
    # "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 21,
    "legend.fontsize": 21,
    "xtick.labelsize": 21,
    "ytick.labelsize": 21,
    "lines.linewidth": 3,
    "lines.markersize": 6,
    "axes.linewidth": 2,
    "figure.figsize": (8, 6),
    "figure.dpi": 400,
})


def amplitude(a, c=0):
    alpha, beta = su2_exp(a, c)
    p = np.abs(beta) ** 2
    phase = - np.angle(alpha)
    return p, phase


def whole_axis_magnus(g_vals, N=10000, which='s'):
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
    g_vals = np.linspace(1e-5, 1.5, 200)

    a1_vals, c2_vals, a3_vals = whole_axis_magnus(g_vals, N=20000, which='s')

    # plt.figure(figsize=(6, 4.5))

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
        plt.xlim(min(g_vals), max(g_vals))
        # plt.yscale('log')
        fig_filename = "Magnus-Landau-Zener-Probability.pdf"

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
        plt.xlim(0, max(g_vals))
        fig_filename = "Magnus-Landau-Zener-Phase.pdf"
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
    # demo('terms')
