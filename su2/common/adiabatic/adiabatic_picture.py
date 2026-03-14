import numpy as np
from scipy.integrate import cumulative_trapezoid

"""


"""


def phi_by_integral(x, omega_func):
    """Compute φ(x) by numerical integration"""

    def _integrand(t):
        return omega_func(t)

    phi_vals = cumulative_trapezoid(_integrand(x), x, initial=0.0)
    phi_vals = np.array(phi_vals)

    # fix gauge by setting φ(0) = 0
    phi0 = np.interp(0, x, phi_vals)
    phi_vals -= phi0
    return phi_vals


def eta_by_integral(t, E_func, omega_func, phi_func=None):
    """
    Compute η(t) = -i * [dθ̇(t) / 2] * exp(2 i φ(t))

    where dθ̇(t) = E(t) / ω(t)^2
    """
    if phi_func is None:
        phi_vals = phi_by_integral(t, omega_func)
    else:
        phi_vals = phi_func(t)

    theta_d_vals = E_func(t) / (omega_func(t) ** 2)
    return - 1j * theta_d_vals * np.exp(2j * phi_vals) / 2
