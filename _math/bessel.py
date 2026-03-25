from scipy.special import jv
import numpy as np
from scipy.integrate import quad


def demo():
    x = np.linspace(-5, 5, 1000)
    y = jv(0, 1j * x)

    print(jv(0, 0))
    import matplotlib.pyplot as plt
    plt.plot(x, y.real, label='Re J0(ix)')
    plt.plot(x, y.imag, label='Im J0(ix)')
    plt.title('Bessel function of the first kind (order 0)')
    plt.xlabel('x')
    plt.ylabel('J0(x)')
    plt.legend()
    plt.grid()
    plt.show()


def integral():
    g = 1.567j

    def integrand(t):
        return np.exp(1j * g * np.sin(t)) / (2 * np.pi)

    def real_integrand(t):
        return integrand(t).real

    def imag_integrand(t):
        return integrand(t).imag

    res_real, _ = quad(real_integrand, 0, 2 * np.pi)
    res_imag, _ = quad(imag_integrand, 0, 2 * np.pi)
    result = res_real + 1j * res_imag

    print(result - jv(0, g))  # Should be close to zero


if __name__ == "__main__":
    demo()

