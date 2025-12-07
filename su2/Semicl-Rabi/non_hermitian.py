import numpy as np

from exact_solution import quasi_energy
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 14


def demo_non_hermitian():
    f0, f1 = 0, 3
    f_vals = 1j * np.linspace(f0, f1, 201)
    v = 0.2

    e_vals = np.array([quasi_energy(f, v, real_only=False) for f in f_vals])
    re_e = e_vals.real
    im_e = e_vals.imag

    # find the exceptional point
    idx = np.argmin(np.abs(im_e[::-1]))
    f_ep = f_vals[::-1][idx].imag

    line1 = plt.plot(f_vals.imag, re_e, label=r'Re$(\epsilon)$')
    color1 = line1[0].get_color()
    plt.plot(f_vals.imag, -re_e - 1, color=color1)
    plt.plot(f_vals.imag, re_e + 1, color=color1)
    plt.plot(f_vals.imag, -re_e, color=color1)

    line2 = plt.plot(f_vals.imag, im_e, '--', label=r'Im$(\epsilon)$')
    plt.plot(f_vals.imag, -im_e, '--', color=line2[0].get_color())

    plt.plot([f_ep, f_ep], [-1, 1], ':', color='gray')
    # plt.plot([f_ep], [-1], 'o', color='black')
    plt.text(f_ep + 0.05, -1+0.05, 'Exceptional Point', color='black')

    plt.xlim(f0, f1)
    plt.ylim(-1, 1)
    plt.xlabel('Im(A)', fontsize=14)
    plt.ylabel(r'$\epsilon$', fontsize=14)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(f'figures/rabi/non-hermitian_demo_d={v:.1f}.pdf', dpi=400)
    plt.show()


if __name__ == '__main__':
    demo_non_hermitian()