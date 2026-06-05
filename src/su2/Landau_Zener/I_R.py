import numpy as np
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


"""

"""

def g_s(s):
    return 0.5 * np.sinh(2 * s) + s


def IR_r(r, cutoff=20, dt=1e-5):
    """
    Computes I_R(r) using trapezoidal rule from scipy.integrate.trapezoid.
    """
    s = np.arange(0, cutoff, dt)
    sh = np.sinh(s)
    ch = np.cosh(s)
    g = g_s(s)
    term1 = (r / ch) * np.sin(r * sh)
    term2 = (2 * sh / ch ** 3) * np.cos(r * sh)
    integrand = (np.sin(g) * (term1 + term2)) / np.pi
    return trapezoid(integrand, s)


def IR_asymp(r):
    """
    Asymptotic form of I_R(r) for large r.
    """
    return 2 / (r * np.sqrt(np.pi)) * np.cos(r ** 2 / 4 - np.pi / 4)


def show():
    """"""
    r_values = np.linspace(200, 400, 100)
    I_values = np.array([IR_r(r) for r in tqdm(r_values)])

    plt.figure(figsize=(8, 5))
    plt.plot(r_values, r_values*I_values, label=r'$rI(r)$')

    plt.xlabel('r')
    plt.ylabel('I(r)')
    plt.title(r'$I(r)$ computed with scipy.integrate.trapezoid', fontsize=13)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()


def compare(recompute=False):
    data_file = '../Sauter-Schwinger/double-sauter/data/IR_asymp@r=20.npz'
    if recompute or not os.path.exists(data_file):
        r_values = np.linspace(20, 22, 200)
        I_values = np.array([IR_r(r) for r in tqdm(r_values)])
        np.savez(data_file, r=r_values, I=I_values)
    else:
        data = np.load(data_file)
        r_values = data['r']
        I_values = data['I']

    I_values_asymp = IR_asymp(r_values)

    plt.figure(figsize=(7, 5))
    plt.plot(r_values, I_values, label=r'numeric')
    plt.plot(r_values, I_values_asymp, '--', label=r'asymptotic')

    plt.xlabel('r')
    plt.ylabel('I(r)')
    # plt.title(r'$I_R(r)$ numeric vs asymptotic', fontsize=13)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


if __name__ == '__main__':
    # show()
    compare(True)