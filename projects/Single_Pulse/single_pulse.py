import numpy as np

from su2.common.su2_integrator import u
from adiabatic_picture import v_func_num
from su2.common.magnus import a1_integral, c2_integral, a3_integral
from su2.common import su2_exp


"""
Single pulse model

H = 1/2 * [[d, f(t)],
           [f(t), d]]
           
where f(t) vanishes at t->+-infinity.

Last updated: 2026-Mar-23
"""


class SinglePulse:
    def __init__(self, f_func, f_dot_func, *args, **kwargs):
        self.f_func = f_func
        self.f_dot_func = f_dot_func




def p_magnus_approx(f, f_dot, d, *args, t_lim=20, N=1000, order=1):
    ts = np.linspace(-t_lim, t_lim, N)
    v_vals = v_func_num(ts, f, f_dot, d, args)

    A1 = a1_integral(v_vals, ts[0], ts[-1], N=N)
    C2 = c2_integral(v_vals, ts[0], ts[-1], N=N)
    A3 = a3_integral(v_vals, ts[0], ts[-1], N=N)

    _, beta_1st = su2_exp(A1, 0)
    p_1st = np.abs(beta_1st) ** 2
    if order == 1:
        return p_1st, None, None

    _, beta_2nd = su2_exp(A1, C2)
    p_2nd = np.abs(beta_2nd) ** 2
    if order == 2:
        return p_1st, p_2nd, None

    _, beta_3rd = su2_exp(A1+A3, C2)
    p_3rd = np.abs(beta_3rd) ** 2
    return p_1st, p_2nd, p_3rd


def p_final(f_func, d, *args, t_lim=20, Nt=1000, psi_0=np.array([0, 1])):
    """Numerical transition probability solved in Schrödinger picture"""

    def h(t):
        return np.array([f_func(t, *args), 0, d])

    ut = u(t_lim, -t_lim, Nt, h)
    amp_final = psi_0.conjugate() @ (ut @ psi_0)
    return 1 - np.abs(amp_final) ** 2
