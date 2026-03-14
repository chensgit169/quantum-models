import numpy as np
from tqdm import tqdm

from su2.common.math.heuc import heun_p, heun_m

N = 1000  # default number of terms in series expansion


def z(t):
    return np.sin(t / 2) ** 2


def psi_1(t, f, v):
    # TODO: connect to t = pi
    zs = z(t)
    psi = np.exp(1j * f * zs) * heun_p(f, v, zs, N)
    return psi


def psi_2(t, f, v):
    zs = z(t)
    psi = -1j * v * np.exp(-1j * f * zs) * zs ** (1 / 2) * heun_m(f, v, zs, N).conjugate()
    return psi


def eta_p(f, v):
    return heun_p(f, v, 1 / 2, n=N)


def eta_m(f, v):
    return heun_m(f, v, 1 / 2, n=N)


def r(g, d):
    # TODO: explain
    eta_p_val = eta_p(g, d)
    eta_m_val = eta_m(g, d)
    _re = np.real(np.exp(1j * g) * eta_p_val * eta_m_val)
    return 2 ** (1 / 2) * d * _re


def quasi_energy(g, d, real_only=True):
    _x = r(g, d)
    if real_only:
        e = np.arcsin(_x) / np.pi
    else:
        e = np.arcsin(_x.astype(np.complex128)) / np.pi
    return e


def demo():
    import matplotlib.pyplot as plt

    t = np.linspace(0, 0.99 * np.pi, 1001)
    f = 1 / 2
    v = 1

    psi = psi_2(t, f, v)
    re = np.real(psi)
    im = np.imag(psi)
    plt.plot(t, re, label='Re')
    plt.plot(t, im, label='Im')
    plt.plot(t, np.zeros_like(t), '--', color='gray')
    plt.xlabel('t')
    plt.ylabel(r'$\psi_1(t)$')
    plt.title(f'f={f}, v={v}')
    plt.legend()
    plt.show()


def demo_quasi_energy():
    import matplotlib.pyplot as plt
    from scipy.special import jv

    d = 0.9
    g_vals = np.linspace(0.001, 10, 400)

    eps = np.array([quasi_energy(g, d) for g in tqdm(g_vals)])
    j0 = jv(0, g_vals)
    approx = d * j0 / 2

    plt.figure(figsize=(7, 5))
    plt.plot(g_vals, eps, label='exact')
    plt.plot(g_vals, approx, label=r'$\frac{\Delta}{2} J_0(f)$', linestyle='--')

    plt.xlabel('f')
    plt.ylabel(r'$\epsilon$')
    plt.title(f'Quasi-energy in Rabi Model d={d}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def quasienergy_3d_plot():
    import matplotlib.pyplot as plt

    g_vals = np.linspace(0.01, 20, 200)
    d_vals = np.linspace(0.01, 20, 200)

    G, D = np.meshgrid(g_vals, d_vals)
    Eps = np.zeros_like(G)

    for i in range(G.shape[0]):
        for j in range(G.shape[1]):
            Eps[i, j] = quasi_energy(G[i, j], D[i, j])

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(G, D, Eps, cmap='viridis')

    ax.set_xlabel('g')
    ax.set_ylabel('d')
    ax.set_zlabel(r'$\epsilon$')
    ax.set_title('Quasi-energy in Rabi Model')

    plt.show()


if __name__ == '__main__':
    # demo_quasi_energy()
    quasienergy_3d_plot()