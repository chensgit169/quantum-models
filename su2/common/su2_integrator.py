import numpy as np
from typing import Callable
from su2.common.pauli import s0, s

"""
Two-level integrator based on analytic formula for su(2) exponent.

last modified: 2026-Mar-11
Chen
"""


def sinx_over_x(x):
    """Compute sin(x)/x safely for small x."""

    x = np.asarray(x)

    res = np.ones_like(x, dtype=float)

    mask = np.abs(x) > 1e-8  # sinx/x = 1 - x^2/6 + x^4/120 + O(x^6)
    res[mask] = np.sin(x[mask]) / x[mask]

    return res


def u_dt(t, dt, h: Callable, *args, order: int = 2):
    """
    Compute time-evolution u(t+dt, dt) for

        H(t) = h(t) @ sigma

    using Magnus expansion with analytic SU(2) exponent.

    order : int
        2 → Magnus midpoint (2nd-order integrator)
        4 → Magnus 4th-order integrator
    """

    if order == 2:

        ht = h(t + dt / 2, *args)

        theta = dt * np.linalg.norm(ht)

        h_sigma = np.einsum('i,ijk->jk', ht, s)

        return np.cos(theta) * s0 - 1j * dt * sinx_over_x(theta) * h_sigma

    elif order == 4:

        c1 = 0.5 - np.sqrt(3) / 6
        c2 = 0.5 + np.sqrt(3) / 6

        h1 = h(t + c1 * dt, *args)
        h2 = h(t + c2 * dt, *args)

        g = (
                0.5 * dt * (h1 + h2)
                + (np.sqrt(3) / 6) * dt ** 2 * np.cross(h2, h1)
        )

        theta = np.linalg.norm(g)

        g_sigma = np.einsum('i,ijk->jk', g, s)

        return np.cos(theta) * s0 - 1j * sinx_over_x(theta) * g_sigma

    else:
        raise ValueError(f"order must be 2 or 4, got {order}")


def u(t, t0, N, h, *args, order=4):
    dt = (t - t0) / N

    ut = s0.copy()
    tk = t0

    for _ in range(N):
        ut = u_dt(tk, dt, h, *args, order=order) @ ut
        tk += dt

    return ut


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
    theta = omega * (t - t0)

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
