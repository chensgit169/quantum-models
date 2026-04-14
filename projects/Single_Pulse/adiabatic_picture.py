from typing import Callable, Union

import numpy as np
from scipy.integrate import cumulative_trapezoid

"""
H(t)=1/2[d(t)σ_z+f(t)σ_x] ->

H(t) = 1/2 [
"""


def instantaneous_eigenstates(chi):
    cos = np.cos(chi / 2)
    sin = np.sin(chi / 2)

    psi_p = np.array([cos, sin])
    psi_m = np.array([-sin, cos])
    return psi_p, psi_m


def _evaluate(x, y: Union[float, np.ndarray, Callable], *arg):
    if callable(y):
        return np.array(y(x, *arg))
    else:
        return np.array(y)  # float or array


def chi_dot(ts: np.ndarray,
            f: Union[float, np.ndarray, Callable],
            f_dot: Union[float, np.ndarray, Callable],
            d: Union[float, np.ndarray, Callable],
            *arg,
            d_dot: Union[float, np.ndarray, Callable] = 0,
            ) -> np.ndarray:
    """
    Compute χ̇(t) = [ḟ(t) * d(t) - ḋ(t) * f(t)] / [d(t)^2 + f(t)^2],
    where tan(χ(t)) = f(t) / d(t)
    """
    fp_vals = _evaluate(ts, f_dot, *arg)
    dp_vals = _evaluate(ts, d_dot, *arg)
    d_vals = _evaluate(ts, d, *arg)
    f_vals = _evaluate(ts, f, *arg)

    eta_vals = (fp_vals * d_vals - dp_vals * f_vals) / (d_vals ** 2 + f_vals ** 2 + 1e-15)

    return eta_vals


def omega(ts: np.ndarray,
          f: Union[float, np.ndarray, Callable],
          d: Union[float, np.ndarray, Callable],
          *arg) -> np.ndarray:
    """
        Compute ω(t) = sqrt(d(t)^2 + f(t)^2) / 2
    """
    f_vals = _evaluate(ts, f, *arg)
    d_vals = _evaluate(ts, d, *arg)

    omega_vals = np.sqrt(d_vals ** 2 + f_vals ** 2) / 2

    return omega_vals


def phi_num(ts: np.ndarray,
            f: Union[float, np.ndarray, Callable],
            d: Union[float, np.ndarray, Callable],
            *arg,
            t0: float = 0.0
            ) -> np.ndarray:
    """ Compute

            φ(x) = ∫_t0^t ω(t) dt

        by numerical integration"""

    omega_vals = omega(ts, f, d, *arg)

    phi_vals = cumulative_trapezoid(omega_vals, ts, initial=0.0)
    phi_vals = np.array(phi_vals)

    # fix gauge by setting φ(t_0) = 0
    phi0 = np.interp(t0, ts, phi_vals)
    phi_vals -= phi0
    return phi_vals


def v_func_num(ts: np.ndarray,
               f: Union[float, np.ndarray, Callable],
               f_dot: Union[float, np.ndarray, Callable],
               d: Union[float, np.ndarray, Callable],
               *arg,
               d_dot: Union[float, np.ndarray, Callable] = 0,
               ) -> np.ndarray:
    """
        Compute v(t) = i * χ̇(t) * exp(2i φ(t)) / 2
    """

    chi_dot_vals = chi_dot(ts, f, f_dot, d, *arg, d_dot=d_dot)
    phi_vals = phi_num(ts, f, d, *arg)
    v_vals = 1j * chi_dot_vals * np.exp(2j * phi_vals) / 2
    return v_vals


def h_num(ts: np.ndarray,
          f: Union[float, np.ndarray, Callable],
          f_dot: Union[float, np.ndarray, Callable],
          d: Union[float, np.ndarray, Callable],
          *arg,
          d_dot: Union[float, np.ndarray, Callable] = 0,
          ) -> np.ndarray:
    """
    Compute η(t) = i * χ̇(t) * exp(2i φ(t)) / 2
    """
    v_vals = 2 * v_func_num(ts, f, f_dot, d, *arg, d_dot=d_dot)

    hx = v_vals.real
    hy = - v_vals.imag
    hz = np.zeros_like(hx)

    return np.stack([hx, hy, hz], axis=-1)
