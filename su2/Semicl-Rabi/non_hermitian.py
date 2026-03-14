import numpy as np

from exact_solution import quasi_energy
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14

"""
Replace g -> ig in the Rabi model, and see the exceptional point 
where the quasienergy becomes complex.
"""


def demo_non_hermitian():
    g0, g1 = 0, 1
    g_vals = 1j * np.linspace(g0, g1, 201)
    d = 0.5

    e_vals = np.array([quasi_energy(g, d, real_only=False) for g in g_vals])
    re_e = e_vals.real
    im_e = e_vals.imag

    # find the exceptional point
    idx = np.argmin(np.abs(im_e[::-1]))
    f_ep = g_vals[::-1][idx].imag

    line1 = plt.plot(g_vals.imag, re_e, label=r'Re$(\epsilon)$')
    color1 = line1[0].get_color()
    plt.plot(g_vals.imag, -re_e - 1, color=color1)
    plt.plot(g_vals.imag, re_e + 1, color=color1)
    plt.plot(g_vals.imag, -re_e, color=color1)

    line2 = plt.plot(g_vals.imag, im_e, '--', label=r'Im$(\epsilon)$')
    plt.plot(g_vals.imag, -im_e, '--', color=line2[0].get_color())

    plt.plot([f_ep, f_ep], [-1, 1], ':', color='gray')
    # plt.plot([f_ep], [-1], 'o', color='black')
    plt.text(f_ep + 0.05, -1+0.05, 'Exceptional \nPoint', color='black')

    plt.xlim(g0, g1)
    plt.ylim(-1, 1)
    plt.xlabel('Im(g)', fontsize=14)
    plt.ylabel(r'$\epsilon$', fontsize=14)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(f'figures/rabi/non-hermitian_demo_d={d:.1f}.pdf', dpi=400)
    plt.show()


if __name__ == '__main__':
    demo_non_hermitian()
