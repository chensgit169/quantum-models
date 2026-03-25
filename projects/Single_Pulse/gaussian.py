import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from single_pulse import p_final, p_magnus_approx


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
    d_vals = np.linspace(0, 10, 20)
    g_vals = np.linspace(0, 40, 200)

    p_exact = np.array([[p_final(f_gauss, d, g) for g in g_vals] for d in tqdm(d_vals)])
    # p_exact = np.array([[p_magnus_approx(f_gauss, f_dot_gauss, d, g, order=1)[0] for g in g_vals] for d in tqdm(d_vals)])

    np.savez(data_file, d_vals=d_vals, g_vals=g_vals, p_exact=p_exact)

    plt.figure(figsize=(8, 6))
    xs, ys = np.meshgrid(g_vals, d_vals)
    plt.contourf(xs, ys, p_exact, levels=50, cmap='viridis')
    plt.colorbar(label='Transition Probability')
    plt.xlabel(r'$g$')
    plt.ylabel(r'$d$')
    plt.title('Transition Probability vs g and d (Gaussian Pulse)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def demo_exact():
    """
    demonstrate the dependence of transition probability on pulse area g,
    """
    d = 10

    g_vals = np.linspace(40, 80, 300)

    p_numerical = np.array([p_final(f_gauss, d, g) for g in tqdm(g_vals)])
    p_1st, p_2nd, p_3rd = np.array([p_magnus_approx(f_gauss, f_dot_gauss, d, g, order=3) for g in tqdm(g_vals)]).T

    plt.plot(g_vals, p_numerical, 'o', label='Numerical', color='red')
    plt.plot(g_vals, p_1st, '--', label='Magnus 1st', color='blue')
    plt.plot(g_vals, p_2nd, ':', label='Magnus 2nd', color='green')
    plt.plot(g_vals, p_3rd, '-', label='Magnus 3rd', color='orange')

    plt.xlim(min(g_vals), max(g_vals))
    plt.xlabel(r'$g$')
    plt.ylabel(r'$P$')
    plt.title(f'Transition Probability vs g, d={d}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    demo_exact()
    # exact_data()