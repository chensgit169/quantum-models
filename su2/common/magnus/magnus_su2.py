from typing import Callable, Union

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid


"""
Magnus expansion to third order for SU(2) Hamiltonian in the form:

                H(t) = [[0, v(t)],
                        [v*(t), 0]] 

where v(t) is a complex-valued function.


Last updated: 2026-Mar-23
"""


def _evaluate(x, y: Union[np.ndarray, Callable], *arg):
    if callable(y):
        return np.array(y(x, *arg))
    else:
        return np.array(y)  # float or array


def magnus_su2(v_func, ti, tf, *args, N=400):
    ts = np.linspace(ti, tf, N)
    v_vals = _evaluate(ts, v_func, *args)


    A1_t = cumulative_trapezoid(v_vals, ts, initial=0)  # A1(t) = ∫ v(t') dt'
    A1 = A1_t[-1]  # A1 = A1(tf)

    c2_1 = 2 * np.imag(v_vals.conj() * A1)
    C2_t = -1/2 * cumulative_trapezoid(c2_1, ts, initial=0)  # C2(t) = B1 ∫ Im[v*(t') A1(t')] dt'
    C2 = C2_t[-1]  # C2 = C2(tf)



def a1_integral(v: Union[np.ndarray, Callable], ti, tf, *args, N=400):
    t = np.linspace(ti, tf, N)
    v_vals = _evaluate(t, v, *args)

    return trapezoid(v_vals, t)


def c2_integral(v: Union[np.ndarray, Callable], ti, tf, *args, N=400):
    t = np.linspace(ti, tf, N)
    f_vals = _evaluate(t, v, *args)

    F = cumulative_trapezoid(f_vals, t, initial=0)  # ∫ f
    F_conj = np.conj(F)  # ∫ f*

    integrand = np.imag(f_vals * F_conj)
    total = trapezoid(integrand, t)

    return total


def a3_integral(v: Union[np.ndarray, Callable], ti, tf, *args, N=400):
    """
    Computes the triple integral for the third-order Magnus expansion term:

        ∫∫∫[ f(t1)f(t2)f*(t3) + f*(t1)f(t2)f(t3) - 2 f(t1)f*(t2)f(t3) ] / 3,

    with the integration limits ti <= t3 <= t2 <= t1 <= tf.
    """
    ts = np.linspace(ti, tf, N)
    v_vals = _evaluate(ts, v, *args)
    f_conj = np.conjugate(v_vals)

    # Precompute outer and middle integrals using cumulative trapezoids
    # cumulative_trapezoid computes ∫ f dt from t[0] up to each point.
    F = cumulative_trapezoid(v_vals, ts, initial=0)  # ∫ f
    F_conj = np.conj(F)  # ∫ f*

    # We will build a matrix for integrals over t3 for each (t2, t1)
    # Outer integration variable: t1 indices
    integral_vals = np.zeros_like(ts, dtype=complex)

    # Vectorized over t1: for each t1 index i1, consider all t2 < t1
    for i1 in range(1, N):
        f_t1 = v_vals[i1]
        f_conj_t1 = f_conj[i1]

        # t2 values and function values up to i1
        t2 = ts[:i1 + 1]
        f_t2 = v_vals[:i1 + 1]

        # Precompute cumulative integrals over t3 for f and f* up to each t2
        I_f = F[:i1 + 1]  # ∫ f(t3) dt3 from 0 to t2
        I_f_conj = F_conj[:i1 + 1]

        # Using vectorization for the two terms:
        # Term1: 2i * f(t1) * Im[f(t2) * ∫f*(t3)dt3]
        term1 = 2j * f_t1 * np.imag(f_t2 * I_f_conj)
        # Term2: 2i * Im[f*(t1) * f(t2)] * ∫f(t3)dt3
        term2 = 2j * np.imag(f_conj_t1 * f_t2) * I_f

        # Combine terms for all t2 points under t1
        integrand_t2 = (term1 + term2) / 3

        # Integrate over t2
        integral_vals[i1] = cumulative_trapezoid(integrand_t2, t2, initial=0)[-1]

    # Integrate over t1
    return trapezoid(integral_vals, ts)
