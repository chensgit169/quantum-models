import numpy as np
from typing import Callable
from scipy.interpolate import interp1d

from su2.common.pauli import s0, s
from su2.common.utils import sinx_over_x

"""
Two-level integrator based on the Magnus expansion and analytic formula for su(2) exponent.

last modified: 2026-Mar-18
Chen
"""


def u_dt(t, dt, h: Callable, *args, order: int = 4):
    """
    Compute time-evolution u(t+dt, dt) for

        H(t) = 1/2 * h(t) @ sigma

    using Magnus expansion with analytic SU(2) exponent.

    order : int
        2 → Magnus midpoint (2nd-order integrator)
        4 → Magnus 4th-order integrator
    """

    if order == 2:

        gt = h(t + dt / 2, *args) / 2

    elif order == 4:

        c1 = 0.5 - np.sqrt(3) / 6
        c2 = 0.5 + np.sqrt(3) / 6

        g1 = h(t + c1 * dt, *args) / 2
        g2 = h(t + c2 * dt, *args) / 2

        gt = 0.5 * dt * (g1 + g2) + (np.sqrt(3) / 6) * dt ** 2 * np.cross(g2, g1)

    else:
        raise ValueError(f"order must be 2 or 4, got {order}")

    theta = np.linalg.norm(gt)

    g_sigma = np.einsum('i,ijk->jk', gt, s)

    return np.cos(theta) * s0 - 1j * sinx_over_x(theta) * g_sigma


def evolve(psi0, t, t0, N, h, *args, order=4):
    psi_t = []
    psi = psi0.copy()

    dt = (t - t0) / N
    tk = t0

    for _ in range(N):
        psi = u_dt(tk, dt, h, *args, order=order) @ psi
        tk += dt
        psi_t.append(psi)

    return np.array(psi_t)


def u(t, t0, N, h, *args, order=4):
    dt = (t - t0) / N

    ut = s0.copy()
    tk = t0

    for _ in range(N):
        ut = u_dt(tk, dt, h, *args, order=order) @ ut
        tk += dt

    return ut


def make_h_interp(t_data, h_data, kind='cubic'):
    """
    Create callable h(t) based on given data points (t_data, h_data) via interpolation.
    """
    interp = interp1d(t_data, h_data, axis=0, kind=kind, fill_value="extrapolate")

    def h(t, *args):
        return interp(t)

    return h


def demo():
    """
    Demonstration of the SU(2) Magnus propagator.
    Tests a constant Hamiltonian H = omega * sigma_z.
    """

    # Hamiltonian: H = omega * sigma_z
    def h(t, omega):
        return np.array([0.0, 0.0, omega])

    # parameters
    omega = 1.0
    t0 = 0.0
    t = 2.0
    N = 100

    # numerical propagator
    U_num = u(t, t0, N, h, omega)

    # exact propagator
    theta = omega * (t - t0) / 2

    U_exact = np.array([
        [np.exp(-1j * theta), 0],
        [0, np.exp(1j * theta)]
    ])

    # error
    err = np.linalg.norm(U_num - U_exact)
    print("Error ||U_num - U_exact|| =", err)

    # unitarity check
    print("Error ||U†U - I|| =",
          np.linalg.norm(U_num.conj().T @ U_num - s0))


if __name__ == "__main__":
    demo()
