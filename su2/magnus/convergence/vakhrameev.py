import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from blanes import load_bn, b_n


"""


"""

bn_list = load_bn()


def integrand(x):
    if np.abs(x) < 1e-8:  # limit x->0
        return 1
    else:
        return 1 / (2 + x * (1 - 1 / np.tan(x)))


def I(s):
    res = quad(lambda x: integrand(x), 0, s)
    return res[0]


# g(x) is the inverse function of I(s)
def g_inv(gs):
    return np.array([I(s) for s in gs])


def g_poly(x, n):
    coeffs = [b_n(k) for k in range(n+1)][::-1]
    p = np.poly1d(coeffs)
    return p(x)


def g_poly_odd(x, n):
    coeffs = []
    for k in range(n + 1):
        if k % 2 == 1:
            coeffs.append(b_n(k))
        else:
            coeffs.append(0)
    p = np.poly1d(coeffs[::-1])
    return p(x)


def g_poly_even(x, n):
    coeffs = []
    for k in range(n + 1):
        if k % 2 == 0:
            coeffs.append(b_n(k))
        else:
            coeffs.append(0)
    p = np.poly1d(coeffs[::-1])
    return p(x)


g_vals = np.linspace(0, np.pi, 1000)
x_vals = g_inv(g_vals)


def demo_approx():
    x_demo = x_vals

    plt.figure(figsize=(8, 5))
    plt.plot(x_demo, g_vals, '--',  label='g(x)')

    for n in range(1, 7, 2):
        g_approx = g_poly(x_vals, n)
        plt.plot(x_demo, g_approx, label=f'n={n}')

    plt.legend()
    plt.grid(True)
    plt.xlabel('x')
    plt.savefig('figures/vakhrameev_g(x).pdf', dpi=400)
    plt.show()


def demo_odd_even():
    x_demo = x_vals # / np.pi
    n = 10

    g_odd = g_poly_odd(x_vals, n)
    g_even = g_poly_even(x_vals, n)

    plt.figure(figsize=(8, 5))
    plt.plot(x_demo, g_vals, label='g(x)')
    plt.plot(x_demo, g_odd, label=f'g_odd')
    plt.plot(x_demo, g_even, label=f'g_even')
    plt.plot(x_demo, g_odd + g_even, '--', label=f'g_odd + g_even')

    plt.legend()
    plt.grid(True)
    plt.show()


def demo_truncation_error():
    x_demo = x_vals[100:]
    n_max = 4

    g_odd_vals = g_poly_odd(x_demo, 20)
    g_even_vals = g_poly_even(x_demo, 20)

    plt.figure(figsize=(8, 5))
    plt.semilogy(x_demo, g_odd_vals, label='|g_odd(x)|')

    for n in range(1, n_max + 1, 2):
        g_approx = g_poly_odd(x_demo, n)
        error = np.abs(g_approx - g_odd_vals)
        plt.semilogy(x_demo, error, label=f'n={n}')

    for n in range(2, n_max + 1, 2):
        g_approx = g_poly_even(x_demo, n)
        error = np.abs(g_approx - g_even_vals)
        plt.semilogy(x_demo, error, '--', label=f'n={n}')

    plt.legend()
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('Truncation Error')
    plt.savefig('figures/vakhrameev_truncation_error.pdf', dpi=400)
    plt.show()


if __name__ == '__main__':
    # demo_approx()
    # demo_odd_even()
    demo_truncation_error()
