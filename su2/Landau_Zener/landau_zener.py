from su2.common.su2_integrator import u_dt
from su2.common.pauli import s0, sx

import numpy as np
import matplotlib.pyplot as plt

"""
Landau-Zener model solved in Schrödinger picture

H = 1/2 * [[v*t, d],
           [d, -v*t]]


"""


# initial state is the eigen state of a*t
psi_0 = np.array([0, 1])


def h_landau_zener(t, a, d) -> np.ndarray:
    # Landau-Zener Hamiltonian
    return np.array([d, 0, a * t]) / 2


def landau_zener_formula(v, d):
    alpha = d ** 2 / (2 * v)  # Landau-Zener parameter

    return np.exp(-np.pi * alpha)


def symmetry():
    a = 0.3
    d = 1

    ut = s0

    # compute evolution operator U(T_lim, 0)
    dt = 0.01
    t_lim = 200
    ts = np.arange(0, t_lim, dt)
    for t in ts:
        ut = u_dt(t, dt, h_landau_zener, a, d) @ ut

    # use symmetry U(0, -T)=P U(T,0)^* P
    uf = ut @ sx @ ut.conjugate() @ sx
    amp_final = psi_0.conjugate() @ (uf @ psi_0)
    p_final = np.abs(amp_final) ** 2

    print("final probability to stay: ", p_final,
          ", Landau-Zener formula: ", landau_zener_formula(a, d))


def demo():
    """
    demonstrate , i.e.
    """
    a = 0.1
    d = 1

    ut = s0

    # compute evolution operator U(T_lim, 0)
    p_t = []
    dt = 0.01
    t_lim = 500
    ts = np.arange(-t_lim, t_lim, dt)
    for t in ts:
        ut = u_dt(t, dt, h_landau_zener, a, d) @ ut
        amp = psi_0.conjugate() @ (ut @ psi_0)
        p = np.abs(amp) ** 2
        p_t.append(p)

    p_t = np.array(p_t)
    p_final = p_t[-1]

    print("final probability to stay: ", p_final,
          ", Landau-Zener formula: ", landau_zener_formula(a, d))
    selected = (30 > ts) & (ts > 0)
    plt.plot(ts[selected], p_t[selected])
    plt.show()


if __name__ == '__main__':
    demo()
