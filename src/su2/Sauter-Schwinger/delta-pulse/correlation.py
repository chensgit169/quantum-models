import numpy as np
from scipy.special import k0, k1


def pi_00(sx, sy, sz):
    """
    Pi_00 = -(|sx|^2 + |sy|^2 + |sz|^2) / 2
    """
    return -(np.abs(sx)**2 + np.abs(sy)**2 + np.abs(sz)**2) / 2


def pi_11(sx, sy, sz):
    """
    Pi_11 = (|sx|^2 + |sy|^2 - |sz|^2) / 2
    """
    return -(np.abs(sx)**2 - np.abs(sy)**2 - np.abs(sz)**2) / 2


def s0(r, m=1.0):
    sx = 1j * k1(m * r) / np.pi
    sy = 0
    sz = k0(m * r) / np.pi

    return sx, sy, sz


def correlations(r, I_R, ip_I_R, I_I, I_beta, ip_I_beta):
    sx_0, sy_0, sz_0 = s0(r)
    sx = sx_0 + ip_I_beta + I_R
    sy = sy_0 + I_I
    sz = sz_0 - I_beta + ip_I_R

    charge_cr = pi_00(sx, sy, sz)
    current_cr = pi_11(sx, sy, sz)
    return charge_cr, current_cr


def demo_free_field():
    r = np.linspace(0.02, 1.2, 100)
    pi00, pi11 = correlations(r, 0, 0, 0, 0, 0)

    import matplotlib.pyplot as plt
    plt.rcParams['font.size'] = 16

    plt.figure(figsize=(8, 6))
    plt.plot(r, -pi00, label=r'$-\Pi_{00}$')
    plt.plot(r, -pi11, label=r'$-\Pi_{11}$', linestyle='--')
    plt.xlabel('|r|')
    # plt.ylabel('Correlation Functions')
    # plt.title('Free Field Correlation Functions')
    plt.legend()
    plt.yscale('log')
    plt.xlim(0, np.max(r))
    plt.grid()
    plt.tight_layout()
    plt.savefig('figures/free_field_correlation.pdf', dpi=400)
    plt.show()




if __name__ == '__main__':
    demo_free_field()
