import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from tqdm import tqdm
import os

from single_pulse import p_final, p_magnus_approx


mpl.rcParams.update({
    # "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 21,
    "legend.fontsize": 16,
    "xtick.labelsize": 21,
    "ytick.labelsize": 21,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "axes.linewidth": 2,
    "figure.figsize": (9, 6.4),  # 单栏尺寸
    "figure.dpi": 400,
})


def f_rz(t, g):
    """Rosen-Zener pulse shape"""
    return g / np.cosh(t)


def f_gauss(t, g):
    """Gaussian pulse shape"""
    return g * np.exp(-t**2)


def f_dot_gauss(t, g):
    """Time derivative of Gaussian pulse shape"""
    return -2 * t * g * np.exp(-t**2)


def exact_data():
    data_file = 'data/gaussian/gaussian_exact_data.npz'

    if os.path.exists(data_file):
        data = np.load(data_file)
        d_vals = data['d_vals']
        g_vals = data['g_vals']
        p_exact = data['p_exact']
    else:
        d_vals = np.linspace(0, 10, 20)
        g_vals = np.linspace(0, 40, 200)

        p_exact = np.array([[p_final(f_gauss, d, g) for g in g_vals] for d in tqdm(d_vals)])
        # p_exact = np.array([[p_magnus_approx(f_gauss, f_dot_gauss, d, g, order=1)[0] for g in g_vals] for d in tqdm(d_vals)])

        np.savez(data_file, d_vals=d_vals, g_vals=g_vals, p_exact=p_exact)

    from matplotlib.colors import LogNorm
    plt.figure(figsize=(8, 6))
    xs, ys = np.meshgrid(g_vals, d_vals)
    plt.contourf(xs, ys, p_exact, levels=50, cmap='viridis', norm=LogNorm())
    plt.colorbar(label='Transition Probability')
    plt.xlabel(r'$\Omega_0 T$')
    plt.ylabel(r'$\Delta T$')

    plt.tight_layout()
    plt.savefig(f'Gaussian/figures/gaussian_P_num.pdf', dpi=400)
    plt.show()


def magnus_data(recompute=True):
    def compute_p(d, g, order=3):
        p = p_magnus_approx(f_gauss, f_dot_gauss, d, g, order=3)[-1]
        return p

    data_file = 'data/gaussian/gaussian_magnus_data.npz'

    if os.path.exists(data_file) and not recompute:
        data = np.load(data_file)
        d_vals = data['d_vals']
        g_vals = data['g_vals']
        p_3rd = data['p_3rd']
    else:
        d_vals = np.linspace(1e-5, 10, 20)
        g_vals = np.linspace(0, 40, 200)

        p_3rd = np.array([[compute_p(d, g, order=3) for g in g_vals] for d in tqdm(d_vals)])

        np.savez(data_file, d_vals=d_vals, g_vals=g_vals, p_3rd=p_3rd)

    from matplotlib.colors import LogNorm

    plt.figure(figsize=(8, 6))
    xs, ys = np.meshgrid(g_vals, d_vals)
    plt.contourf(xs, ys, p_3rd, levels=50, cmap='viridis', norm=LogNorm())
    plt.colorbar(label='Transition Probability')
    plt.xlabel(r'$\Omega_0 T$')
    plt.ylabel(r'$\Delta T$')
    # plt.title('Transition Probability vs g and d (Gaussian Pulse)')
    plt.tight_layout()
    plt.savefig(f'Gaussian/figures/gaussian_P_magnus_3rd.pdf', dpi=400)
    plt.show()


def demo_exact():
    """
    demonstrate the dependence of transition probability on pulse area g,
    """
    d = 3

    g_vals = np.linspace(10, 20, 100)

    p_numerical = np.array([p_final(f_gauss, d, g) for g in tqdm(g_vals)])
    p_1st, p_2nd, p_3rd = np.array([p_magnus_approx(f_gauss, f_dot_gauss, d, g, order=3) for g in tqdm(g_vals)]).T

    plt.plot(g_vals, p_numerical, 'o', label='Exact', color='grey')
    plt.plot(g_vals, p_1st, '--', label='FMA')
    plt.plot(g_vals, p_2nd, ':', label='SMA')
    plt.plot(g_vals, p_3rd, '-.', label='TMA')

    plt.xlim(min(g_vals), max(g_vals))
    plt.xlabel(r'$\Omega_0 T$')
    plt.ylabel(r'Transition Probability $P$')
    # plt.title(f'Transition Probability vs g, d={d}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'Gaussian/figures/gaussian_transition_probability_d={d:.2f}.pdf', dpi=400)
    plt.show()


if __name__ == '__main__':
    demo_exact()
    # exact_data()
    # magnus_data()
