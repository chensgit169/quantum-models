import numpy as np


def check_gamma(gamma):
    if gamma == 0 or (gamma < 0 and abs(gamma - int(gamma)) < 1e-12):
        raise ValueError("γ must not be 0 or a negative integer.")


def check_z(z):
    if np.any(np.abs(z) >= 1):
        raise ValueError("Power series converges only for |z| < 1.")


def recurrence(alpha, delta, gamma, epsilon, q):
    """
    Iterator that generates coefficients {c_k}, k = 0, 1, 2, ...
    determined by the recurrence relation:

        c_{k+1} = [k(k-1 + δ + γ - ε) - q] / [(k+1)(k+γ)] * c_k
                  + [ε(k-1) + α] / [(k+1)(k+γ)] * c_{k-1}

    with initial conditions:
        c_{-1} = 0
        c_0 = 1
    """
    # Validate gamma
    check_gamma(gamma)

    # Initial values
    c_minus1 = 0.0
    c0 = 1.0

    # Yield c_0
    yield c0

    k = 0
    prev = c0

    # Infinite generator
    while True:
        denom = (k + 1) * (k + gamma)
        term1 = (k * (k - 1 + delta + gamma - epsilon) - q) / denom * prev
        term2 = ((epsilon * (k - 1) + alpha) / denom) * c_minus1
        next_val = term1 + term2

        yield next_val

        # Shift values for next iteration
        c_minus1, prev = prev, next_val
        k += 1


def heunC(q, alpha, gamma, delta, epsilon, z, n=100):
    """
    Compute the HeunC function using power series expansion around z=0.
    """
    check_gamma(gamma)
    check_z(z)

    gen = recurrence(alpha, delta, gamma, epsilon, q)
    vals = np.zeros_like(z, dtype=np.complex128)
    for k in range(n):
        ck = next(gen)
        vals += ck * z ** k
    return vals


def param_p(f, v):
    alpha = 2j * f
    delta = 1/2
    gamma = 1/2
    epsilon = 2j * f
    q = 1j * f + v**2 / 4
    return q, alpha, gamma, delta, epsilon


def param_m(f, v):
    alpha = 3j * f
    delta = 1 / 2
    gamma = 3 / 2
    epsilon = 2j * f
    q = 2j * f + (v ** 2 - 1) / 4
    return q, alpha, gamma, delta, epsilon


def heun_p(f, v, z, n=100):
    q, alpha, gamma, delta, epsilon = param_p(f, v)
    return heunC(q, alpha, gamma, delta, epsilon, z, n)


def heun_m(f, v, z, n=100):
    q, alpha, gamma, delta, epsilon = param_m(f, v)
    return heunC(q, alpha, gamma, delta, epsilon, z, n)