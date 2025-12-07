import numpy as np
from scipy.linalg import eigvals, eig

from su2.common.su2_integrator import u
from su2.common.pauli import sz

from scipy.special import j0
import matplotlib.pyplot as plt

import os

from tqdm import tqdm

# frequency used as unit
w = 1
d_vals = np.linspace(0.1, 100, 10)

# parity operator
p = sz

# time-step for integrator
dt = 1e-3


def h(t, d, a):
    return np.array([a * np.sin(w * t), 0, d]) / 2


def quasi_energy(d, a):
    u_T = u(2 * np.pi / w, dt, h, d, a)
    return np.angle(eigvals(u_T)).max()


def quasi_energy_pt(d, a):
    u_T_half = u(np.pi / w, dt, h, d, a)
    u_T = p @ u_T_half @ p @ u_T_half
    # u_T = u(2 * np.pi / w, dt, h, d, a)

    exponents, psis = eig(u_T)[:]

    psi = psis[0]
    e = - np.angle(exponents[0])

    # check PT symmetry of Floquet state
    i = psi.conjugate() @ (p @ u_T_half @ psi) * np.exp(1j*e/2)
    assert (np.abs(i.imag) < 1e-6), f'{np.abs(i.imag)} is too large'

    # select positive PT state
    if np.real(i) > 0:
        return e
    else:
        return - e


def small_d():
    """

    """
    d = 4
    data_path = f'data/e_a_d={d:2f}.npz'

    if os.path.exists(data_path):
        data = np.load(data_path)
        a_vals = data['x']
        es = data['y']
        d = data['d']
    else:
        a_vals = np.linspace(0, 10, 1000)
        es = np.array([quasi_energy_pt(d, a) for a in tqdm(a_vals)])
        np.savez(data_path, x=a_vals, y=es, d=d)

    plt.plot(a_vals, es, '*', label='numerically exact')
    plt.plot(a_vals, np.pi * d * j0(2 * a_vals / w),
             label=r'$\pi \Delta \ J_0(x)$')
    plt.plot(a_vals, np.zeros_like(a_vals), '--')
    plt.xlabel(r'$x=A/\omega$')
    plt.ylabel(r'$\epsilon$')
    # plt.title(r'Quasi-energy for $\Delta$'+f'={d:.2f}')
    plt.legend()
    plt.tight_layout()
    # plt.savefig('figures/d'+f'={d:.2f}.pdf')
    plt.show()


if __name__ == '__main__':
    small_d()