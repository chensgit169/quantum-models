import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import j0
from tqdm import tqdm

from exact_solution import quasi_energy
from su2.common.magnus.magnus_su2 import a1_integral, c2_integral, a3_integral


plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 21


def v(t, g, d):
    return d * np.exp(-1j * g * np.cos(t)) / 2


def demo_exact_eps():
    d = 2
    g_vals = np.linspace(-20, 20, 201)
    e_vals = np.array([quasi_energy(g, d, real_only=False).real for g in g_vals])

    plt.plot(g_vals, e_vals)
    plt.xlabel(r'$g/\omega$')
    plt.ylabel(r'$\epsilon/\omega$')
    # plt.title(r'Quasienergy for $g=$'+f'{g}')
    # plt.xlim(np.min(d_vals), np.max(d_vals))
    # plt.hlines(y=-0.5, xmin=np.min(d_vals), xmax=np.max(d_vals), colors='gray', linestyles='dashed', alpha=0.5)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def direct_magnus():
    d = 1.0
    g_vals = np.linspace(0, 20, 201)
    e_vals = np.array([quasi_energy(f, d, real_only=False).real for f in tqdm(g_vals)])

    line_main = plt.plot(g_vals, e_vals, label='Exact')
    plt.plot(g_vals,  -e_vals, color =line_main[0].get_color())
    # plt.plot(f_vals, e_vals + 1, color=line_main[0].get_color())

    def eps_sg(a_m, c_m):
        theta_m = np.sqrt(np.abs(a_m) ** 2 + np.abs(c_m) ** 2)
        approx = np.arccos(np.cos(theta_m)) / (2 * np.pi)
        return approx

    # compute third order correction

    a1_m = np.array([a1_integral(v, 0, 2 * np.pi, g, d) for g in g_vals])
    c2_m = np.array([c2_integral(v, 0, 2 * np.pi, g, d) for g in g_vals])
    a3_m = np.array([a3_integral(v, 0, 2 * np.pi, g, d) for g in g_vals])

    # approx_1st = d * j0(g_vals) / 2

    approx_1st = eps_sg(a1_m, 0)
    approx_3rd = eps_sg(a1_m + a3_m, c2_m)

    plt.plot(g_vals, approx_1st, '--', label=r'$\Delta \ J_0(A)/2$')
    plt.plot(g_vals, approx_3rd, ':', label=r'3rd Magnus')

    # plt.figure(figsize=(6, 4))
    plt.xlabel(r'$g/\omega$')
    plt.ylabel(r'$\epsilon$')
    plt.xlim(np.min(g_vals), np.max(g_vals))
    # plt.title(r'$\Delta$=' + f'{v}')
    plt.axhline(0, color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    # plt.savefig('figures/limiting/small_gap_demo_error.pdf', dpi=400)
    plt.show()


def magnus_explicit_symmetry(recompute=False):
    d = 1.0
    g_vals = np.linspace(0, 10, 201)
    e_vals = np.array([quasi_energy(f, d, real_only=False).real for f in tqdm(g_vals)])

    plt.figure(figsize=(8, 6))
    # plt.plot(f_vals, e_vals + 1, color=line_main[0].get_color())

    def eps_sg(a_m, c_m):
        theta_m = np.sqrt(np.abs(a_m) ** 2 + np.abs(c_m) ** 2)
        r = np.real(a_m) * np.sin(theta_m) / theta_m
        return np.arcsin(r) / np.pi

    # approximation for small field
    a1_m = np.array([a1_integral(v, 0, np.pi, g, d) for g in g_vals])
    c2_m = np.array([c2_integral(v, 0, np.pi, g, d) for g in g_vals])
    a3_m = np.array([a3_integral(v, 0, np.pi, g, d) for g in g_vals])

    approx_1st = eps_sg(a1_m, 0)
    approx_2nd = eps_sg(a1_m, c2_m)
    approx_3rd = eps_sg(a1_m + a3_m, c2_m)
    # plt.plot(g_vals, approx_1st, '--', label=r'$\frac{\Delta}{2}J_0(\frac{g}{\omega})$ (1st Magnus)')
    # plt.plot(g_vals, approx_2nd, '-.', label=r'2nd Magnus')
    # plt.plot(g_vals, approx_3rd, ':', label=r'3rd Magnus')

    plt.plot(g_vals, approx_1st, '--', label=r'$\frac{\Delta}{2}J_0(\frac{g}{\omega})$ (FMA)')
    plt.plot(g_vals, approx_2nd, ':', label=r'SMA')
    plt.plot(g_vals, approx_3rd, '-.', label=r'TMA')
    # plt.axhline(0, color='gray', linestyle='--')

    line_main = plt.plot(g_vals, e_vals, label='Exact', color='k')
    plt.plot(g_vals, -e_vals, color=line_main[0].get_color())
    # plt.grid(True, alpha=0.3)
    plt.xlim(np.min(g_vals), np.max(g_vals))
    plt.xlabel(r'$g/\omega$', fontsize=21)
    plt.ylabel(r'$\epsilon/\omega$', fontsize=21)
    # plt.title(r'$g=$' + f'{g}')
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/quasienergy/small_gap/"
                +f"explicit_symmetric_d={d}_MA.pdf", dpi=400)
    plt.show()


if __name__ == '__main__':
    # direct_magnus()
    magnus_explicit_symmetry()
    # demo_exact_eps()