import os

import numpy as np
from tqdm import tqdm

from heuc import heun_p, heun_m

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


def r(f, v):
    # TODO: explain
    eta_p_val = eta_p(f, v)
    eta_m_val = eta_m(f, v)
    _re = np.real(np.exp(1j * f) * eta_p_val * eta_m_val)
    return 2 ** (1 / 2) * v * _re


def quasi_energy(f, v, real_only=True):
    _x = r(f, v)
    if real_only:
        e = - np.arcsin(_x) / np.pi
    else:
        e = - np.arcsin(_x.astype(np.complex128)) / np.pi
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
