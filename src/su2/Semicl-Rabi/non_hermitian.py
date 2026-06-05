import numpy as np

from exact_solution import quasi_energy
import matplotlib.pyplot as plt
from scipy.special import jv
from su2.magnus import magnus_su2_complex

plt.rcParams['font.size'] = 14

# TODO: the present calculation diverges

"""
Replace g -> ig in the Rabi model, and see the exceptional point 
where the quasienergy becomes complex.

Last modified: 2026-Mar-18
"""


def demo_bessel():
    g0, g1 = 0, 8
    g_vals = np.linspace(g0, g1, 201)
    d = 0.01

    e_vals = np.array([quasi_energy(1j * g, d, real_only=False) for g in g_vals])
    re_e = e_vals.real

    # exact results
    line1 = plt.plot(g_vals, re_e, label=r'Re$(\epsilon)$')
    color1 = line1[0].get_color()
    plt.plot(g_vals, 1-re_e, color=color1)

    # Bessel
    j0_res = d * jv(0, 1j * g_vals).real / 2
    line_j0 = plt.plot(g_vals, j0_res, '--', label=r'$J_0(ig)$')
    colo2 = line_j0[0].get_color()
    plt.plot(g_vals, 1-j0_res, '--', color=colo2)

    # Magnus
    a1_vals, c2_vals = np.array([magnus(g, d) for g in g_vals]).T
    _temp = np.abs(a1_vals)**2 - np.abs(c2_vals)**2
    eps_magnus = np.sqrt(_temp.astype(complex)) / (2 * np.pi)

    line_magnus = plt.plot(g_vals, eps_magnus.real, ':', label=r'$A_1$')

    plt.xlim(g0, g1)
    plt.ylim(0, 1)
    plt.xlabel('g', fontsize=14)
    plt.ylabel(r'$\epsilon$', fontsize=14)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(f'figures/rabi/non-hermitian_demo_d={d:.1f}.pdf', dpi=400)
    plt.show()


def demo_non_hermitian():
    g0, g1 = 0, 8
    g_vals = np.linspace(g0, g1, 201)
    d = 0.1

    e_vals = np.array([quasi_energy(1j * g, d, real_only=False) for g in g_vals])
    re_e = e_vals.real
    im_e = e_vals.imag

    a1_vals, c2_vals = np.array([magnus(g, d) for g in g_vals]).T / 2

    # find the exceptional point
    idx = np.argmin(np.abs(im_e[::-1]))
    f_ep = g_vals[::-1][idx]

    line1 = plt.plot(g_vals, re_e, label=r'Re$(\epsilon)$')
    color1 = line1[0].get_color()
    plt.plot(g_vals, -re_e - 1, color=color1)
    plt.plot(g_vals, re_e + 1, color=color1)
    plt.plot(g_vals, -re_e, color=color1)

    j0_res = d * jv(0, 1j * g_vals).real / 2
    plt.plot(g_vals, j0_res, '-*', label=r'$J_0(ig)$')

    # plt.plot(g_vals, a1_vals.real, '--', label=r'Re$(a_1)$')
    # plt.plot(g_vals, c2_vals.imag, '--', label=r'Im$(c_2)$')

    line2 = plt.plot(g_vals, im_e, '--', label=r'Im$(\epsilon)$')
    plt.plot(g_vals, -im_e, '--', color=line2[0].get_color())

    plt.plot([f_ep, f_ep], [-1, 1], ':', color='gray')
    # plt.plot([f_ep], [-1], 'o', color='black')
    plt.text(f_ep + 0.05, -1 + 0.05, 'Exceptional \nPoint', color='black')

    plt.xlim(g0, g1)
    plt.ylim(-1, 1)
    plt.xlabel('g', fontsize=14)
    plt.ylabel(r'$\epsilon$', fontsize=14)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(f'figures/rabi/non-hermitian_demo_d={d:.1f}.pdf', dpi=400)
    plt.show()


def v_func(t, g, d):
    return d * np.exp(- g * np.sin(t)) / 2


def u_func(t, g, d):
    return d * np.exp(+ g * np.sin(t)) / 2


def magnus(g, d):
    mc = magnus_su2_complex(v_func, u_func, 0, 2 * np.pi, g, d, N=4000)
    A1 = mc['A1']
    D1 = mc['D1']
    C2 = mc['C2']
    A3 = mc['A3']
    D3 = mc['D3']

    A = A1 + A3
    C = C2

    print(A1, A3)

    print(np.abs(A)-np.abs(C))
    return A, C


if __name__ == '__main__':
    # demo_non_hermitian()
    demo_bessel()
    # magnus(3.5, 0.1)