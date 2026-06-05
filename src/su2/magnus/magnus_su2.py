from typing import Callable, Union

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from sympy import bernoulli
from math import factorial


"""
Magnus expansion to third order for SU(2) Hamiltonian in the form:

                H(t) = [[0, v(t)],
                        [v*(t), 0]] 

where v(t) is a complex-valued function.


Last updated: 2026-Mar-23
"""

b1 = - float(bernoulli(1))  # Bernoulli number B_1 = -1/2, note that in sympy b1=1/2
b2 = float(bernoulli(2))  # Bernoulli number B_2 = 1/6


def _evaluate(x, y: Union[np.ndarray, Callable], *arg):
    if callable(y):
        return np.array(y(x, *arg))
    else:
        return np.array(y)  # float or array


def magnus_su2(v_func, ti, tf, *args, N=400, order=3):
    ts = np.linspace(ti, tf, N)
    a1 = _evaluate(ts, v_func, *args)

    if order not in {1, 2, 3}:
        raise NotImplementedError("Order must be 1, 2, or 3.")

    if order == 1:
        A1_f = trapezoid(a1, ts)
        return A1_f, None, None

    A1 = cumulative_trapezoid(a1, ts, initial=0)  # A1(t) = ∫ v(t') dt'
    A1_f = A1[-1]  # A1 = A1(tf)

    c2_1 = 2 * np.imag(a1.conj() * A1)
    c2 = -1 / 2 * c2_1
    if order == 2:
        C2_f = trapezoid(c2, ts)  # C2 = ∫ c2(t) dt
        return A1_f, C2_f, None

    C2 = cumulative_trapezoid(c2, ts, initial=0)  # C2(t) = B1 ∫ Im[v*(t') A1(t')] dt'
    C2_f = C2[-1]  # C2 = C2(tf)

    # a_3^1 = 2 i C_2 v, a_3^2 = - 2 i A_1 c_2^1
    a3_1 = -2j * C2 * a1
    a3_2 = 2j * A1 * c2_1

    # A_3 = -\frac{1}{2} \int a_3^1 + \frac{1}{12} \int a_3^2.
    a3 = -1 / 2 * a3_1 + 1 / 12 * a3_2
    A3_f = trapezoid(a3, ts)
    return A1_f, C2_f, A3_f


def magnus_su2_complex(v_func, u_func, ti, tf, *args, N=400, order=3):
    ts = np.linspace(ti, tf, N)
    a1 = _evaluate(ts, v_func, *args)
    d1 = _evaluate(ts, u_func, *args)

    res = {}

    if order not in {1, 2, 3}:
        raise NotImplementedError("Order must be 1, 2, or 3.")

    if order == 1:
        res['A1'] = trapezoid(a1, ts)
        res['D1'] = trapezoid(d1, ts)
        return res

    A1 = cumulative_trapezoid(a1, ts, initial=0)  # A1(t) = ∫ v(t') dt'
    D1 = cumulative_trapezoid(d1, ts, initial=0)  # D1(t) = ∫ u(t') dt'
    res['A1'] = A1[-1]
    res['D1'] = D1[-1]

    c2_1 = 1j * (D1 * a1 - A1 * d1)
    c2 = b1 * c2_1
    if order == 2:
        C2_f = trapezoid(c2, ts)  # C2 = ∫ c2(t) dt
        res['C2'] = C2_f
        return res

    C2 = cumulative_trapezoid(c2, ts, initial=0)  # C2(t) = B1 ∫ Im[v*(t') A1(t')] dt'
    C2_f = C2[-1]  # C2 = C2(tf)
    res['C2'] = C2_f

    # a_3^1 = 2 i C_2 v, a_3^2 = - 2 i A_1 c_2^1
    a3_1 = -2j * C2 * a1
    a3_2 = 2j * A1 * c2_1
    a3 = b1 * a3_1 + b2 / factorial(2) * a3_2
    A3_f = trapezoid(a3, ts)

    d3_1 = 2j * C2 * d1
    d3_2 = -2j * D1 * c2_1
    d3 = b1 * d3_1 + b2 / factorial(2) * d3_2
    D3_f = trapezoid(d3, ts)

    res['A3'] = A3_f
    res['D3'] = D3_f

    return res


def _a1_integral(v: Union[np.ndarray, Callable], ti, tf, *args, N=400):
    t = np.linspace(ti, tf, N)
    v_vals = _evaluate(t, v, *args)

    return trapezoid(v_vals, t)


def _c2_integral(v: Union[np.ndarray, Callable], ti, tf, *args, N=400):
    t = np.linspace(ti, tf, N)
    f_vals = _evaluate(t, v, *args)

    F = cumulative_trapezoid(f_vals, t, initial=0)  # ∫ f
    F_conj = np.conj(F)  # ∫ f*

    integrand = np.imag(f_vals * F_conj)
    total = trapezoid(integrand, t)

    return total


def _a3_integral(v: Union[np.ndarray, Callable], ti, tf, *args, N=400):
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


def _check_version():
    ts = np.linspace(0, 10, 20000)
    vs = np.exp(1.2j * ts)
    A1, C2, A3 = magnus_su2(vs, 0, 10, N=20000)
    print('done')
    A1_old = _a1_integral(vs, 0, 10, N=20000)
    print(A1 - A1_old)
    C2_old = _c2_integral(vs, 0, 10, N=20000)
    print(C2 - C2_old)
    A3_old = _a3_integral(vs, 0, 10, N=20000)

    print(A3 - A3_old)


def _check_complex():
    ts = np.linspace(0, 10, 20000)
    vs = np.exp(1.2j * ts)
    us = vs.conjugate()
    A1, C2, A3 = magnus_su2(vs, 0, 10, N=20000)
    print('done')

    res_complex = magnus_su2_complex(vs, us, 0, 10, N=20000, order=3)

    print('done')
    print(res_complex['A1'] - A1)
    print(res_complex['C2'] - C2)
    print(res_complex['A3'] - A3)
