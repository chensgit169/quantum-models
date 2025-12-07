import numpy as np


e1 = 0.25
m = 1
tau2 = 0.51099


def g_complex(x, c):
    """
    c = pi / 2 gamma_c (combined Keldysh parameter)
      = pi * E1 * tau2 / 2
    """
    w = x + 1j * c
    S = np.sqrt(1 + w ** 2)
    return w * S + np.arcsinh(w)  # g(x)


def phi_prime(x, a1, a2, c):
    """Compute Φ'(x) = Im( z'(x) / z(x) )."""
    w = x + 1j * c
    S = np.sqrt(1 + w ** 2)
    g_val = w * S + np.arcsinh(w)

    a1 *= np.exp(-np.pi * m ** 2 / (2 * e1))

    chi2 = np.imag(g_complex(x, c))
    a2 *= np.exp(-np.abs(chi2) * m ** 2 / e1)

    # z(x)
    Ez = np.exp(-1j * g_val)
    z = a1 + a2 * Ez

    # g'(x) = 2 * sqrt(1 + (x+ic)^2) = 2*S
    g_prime = 2 * S

    # z'(x) = -i b g'(x) e^{-ig(x)}
    z_prime = -1j * a2 * g_prime * Ez

    # Φ'(x) = Im(z'(x)/z(x))
    return np.imag(z_prime / z)


# 示例：生成一段 x 的数据
if __name__ == "__main__":

    import matplotlib.pyplot as plt
    a, b, c = 1.4, 0.2, np.pi * e1 * tau2 / 2

    xs = np.linspace(-4, 4, 400)
    phi_vals = phi_prime(xs, a, b, c)

    # 打印前 10 个结果
    plt.plot(xs, phi_vals)
    plt.xlabel('x')
    plt.ylabel("Φ'(x)")
    plt.show()
