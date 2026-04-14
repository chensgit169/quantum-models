import numpy as np
from scipy.special import gamma
from scipy.integrate import cumulative_trapezoid


"""
Landau-Zener Hamiltonian in the adiabatic picture.

Last updated: 2026-Apr-13
"""


def _gamma(v, d):
    g = d**2 / (4 * v)
    return g


def lz_p(g):
    """ Landau-Zener formula for transition probability """
    return np.exp(-2 * np.pi * g)


def stokes_phase(g):
    """
    Exact result of Landau-Zener Stokes phase
        φ_S(g) = π/4 + g(ln g − 1) + arg Γ(1 − ig)
    """
    gamma_val = gamma(1 - 1j * g)
    phi = (np.pi / 4) + g * (np.log(g) - 1) + np.angle(gamma_val)
    return np.mod(phi, 2 * np.pi)  # restrict the phase to [0, 2π)


def x_t(t, v, d):
    return v * t / d


def t_x(x, v, d):
    return x * d / v


# sinh(s) = x

def s_x(x):
    return np.arcsinh(x)


def phi_s(s, g):
    # dynamical phase
    return g * (0.5 * np.sinh(2 * s) + s)


def v_s(s, g):
    # sinh(s) = t
    return 1j * np.exp(2j * phi_s(s, g)) / (2 * np.cosh(s))


def phi_x(x, g):
    # g * (x\sqrt{1+x^2}+\ln|x+\sqrt{1+x^2}|)
    phi_val = g * (x * np.sqrt(1 + x**2) + np.log(np.abs(x + np.sqrt(1 + x**2))))
    return phi_val


def v_x(x, g):
    return 1j * np.exp(2j * phi_x(x, g)) / (2 * (1 + x ** 2))


def h_x(xs: np.ndarray, g) -> np.ndarray:
    """
    Compute η(t) = i * χ̇(t) * exp(2i φ(t)) / 2
    """
    v_vals = v_x(xs, g)

    hx = 2 * v_vals.real
    hy = - 2 * v_vals.imag
    hz = np.zeros_like(hx)

    return np.stack([hx, hy, hz], axis=-1)


def wkb_p(xs, g):
    v_vals = v_x(xs, g)
    a1_vals = cumulative_trapezoid(v_vals, xs, initial=0.0)
    p_1st = np.sin(np.abs(a1_vals))**2
    p_wkb = np.abs(a1_vals)**2
    return p_1st, p_wkb
