import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid
from su2.common.adiabatic.adiabatic_picture import phi_by_integral, eta_by_integral


plt.rcParams['font.size'] = 14


def omega(t):
    return np.sqrt(1 + t ** 2) / 2


def phi(x):
    """
    x = v*t/d
    """
    res = x * np.sqrt(1 + x ** 2) + np.log(x + np.sqrt(1 + x ** 2))
    return res / 4


def h_adiabatic(t, a, d=1):
    """Landau-Zener Hamiltonian under adiabatic picture
    H = [[0, if],
         [-if*, 0]]
    f =
    """
    alpha = d ** 2 / (2 * a)
    f = d * a * np.exp(-1j * alpha * phi(a * t / d)) / (2 * (d ** 2 + (a * t) ** 2))
    hx = - np.imag(f)
    hy = np.real(f)
    return np.array([hx, hy, 0])


def integrand(x, alpha):
    return np.exp(-2j * alpha * phi(x)) / (1 + x ** 2) / 2


def integrand_s(s, alpha):
    # return np.exp(-2j * alpha * phi(np.tan(s))) / 2
    return np.exp(-2j * alpha * phi(np.tan(s)))


def test_phi():
    x_vals = np.linspace(0, 5, 200)
    phi_vals = phi(x_vals)
    phi_int_vals = phi_by_integral(x_vals, omega)

    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, phi_vals, label='Analytical φ(x)')
    plt.plot(x_vals, phi_int_vals, '--', label='Numerical Integration φ(x)')
    plt.xlabel('x')
    plt.ylabel('φ(x)')
    plt.title('Comparison of Analytical and Numerical φ(x)')
    plt.legend()
    plt.grid(True)
    plt.show()


def test_int():
    t_vals = np.linspace(0, 10, 1000)

    def electric(t):
        return 1

    eta_vals = eta_by_integral(t_vals, electric, omega).imag

    a1_vals = cumulative_trapezoid(eta_vals, t_vals, initial=0.0)

    plt.figure(figsize=(8, 5))
    plt.plot(t_vals, a1_vals, label='Numerical Integration of η(t)')
    plt.xlabel('t')
    plt.ylabel('Im[a1(t)]')
    plt.title('Integration of η(t) over time')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # demo_f_integral(9)
    # test_phi()
    # approx_p()
    # demo_approx_p()
    test_int()
