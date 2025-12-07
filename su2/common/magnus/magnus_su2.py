import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid


"""
Magnus expansion to third order for SU(2) Hamiltonian in the form:

                H(t) = [[0, f(t)],
                        [-f*(t), 0]] 

where f(t) is a complex-valued function.


Last updated: Nov. 2th, 2025
"""


def a1_integral(f_func, ti, tf, *args, N=400):
    t = np.linspace(ti, tf, N)
    f_vals = f_func(t, *args)
    return trapezoid(f_vals, t)


def c2_integral(f_func, ti, tf, *args, N=400):
    t = np.linspace(ti, tf, N)
    f_vals = f_func(t, *args)
    f_conj = np.conjugate(f_vals)

    F = cumulative_trapezoid(f_vals, t, initial=0)  # ∫ f
    F_conj = np.conj(F)  # ∫ f*

    integrand = np.imag(f_vals * F_conj)
    total = trapezoid(integrand, t)

    # t1, t2 = np.meshgrid(t, t, indexing='ij')
    #
    # mask = t2 <= t1
    # f_s1 = f_func(t1[mask], *args)
    # f_s2 = f_func(t2[mask], *args)
    #
    # integrand = np.zeros_like(t1)
    # integrand[mask] = np.imag(f_s1 * np.conjugate(f_s2))
    #
    # inner = trapezoid(integrand, t, axis=1)
    # total = trapezoid(inner, t)
    return total


def a3_integral(f_func, ti, tf, *args, N=400):
    """
    Computes the triple integral for the third-order Magnus expansion term:

        ∫∫∫[ f(t1)f(t2)f*(t3) + f*(t1)f(t2)f(t3) - 2 f(t1)f*(t2)f(t3) ] / 3,

    with the integration limits ti <= t3 <= t2 <= t1 <= tf.
    """
    t = np.linspace(ti, tf, N)
    f_vals = f_func(t, *args)
    f_conj = np.conjugate(f_vals)

    # Precompute outer and middle integrals using cumulative trapezoids
    # cumulative_trapezoid computes ∫ f dt from t[0] up to each point.
    F = cumulative_trapezoid(f_vals, t, initial=0)  # ∫ f
    F_conj = np.conj(F)  # ∫ f*

    # We will build a matrix for integrals over t3 for each (t2, t1)
    # Outer integration variable: t1 indices
    integral_vals = np.zeros_like(t, dtype=complex)

    # Vectorized over t1: for each t1 index i1, consider all t2 < t1
    for i1 in range(1, N):
        f_t1 = f_vals[i1]
        f_conj_t1 = f_conj[i1]

        # t2 values and function values up to i1
        t2 = t[:i1 + 1]
        f_t2 = f_vals[:i1 + 1]

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
    return trapezoid(integral_vals, t)
