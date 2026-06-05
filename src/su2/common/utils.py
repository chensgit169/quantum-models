import numpy as np


def sinx_over_x(x):
    """Compute sin(x)/x safely for small x."""

    x = np.asarray(x)

    res = np.ones_like(x, dtype=float)

    mask = np.abs(x) > 1e-8  # sinx/x = 1 - x^2/6 + x^4/120 + O(x^6)
    res[mask] = np.sin(x[mask]) / x[mask]

    return res


def su2_exp(a, c=0):
    """ U = exp(-i Omega),

    U = [ alpha      -beta^* ]
        [ beta       alpha^* ]

    Omega = [  c      a  ]
            [ a^*   -c  ]

    Returns alpha, beta
    """
    theta = np.sqrt(np.abs(a) ** 2 + c ** 2)
    alpha = np.cos(theta) - 1j * sinx_over_x(theta) * c
    beta = -1j * sinx_over_x(theta) * np.conjugate(a)
    return alpha, beta
