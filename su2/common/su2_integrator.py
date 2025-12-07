import numpy as np
from typing import Callable
from su2.common.pauli import s0, s


"""
Two-level integrator based on analytic formula for su(2) exponent.

last modified: 02-11-2025
Chen
"""


def u_dt(t, dt, h: Callable, *args):
    """
    Compute time-evolution u(t+dt, dt) for H(t) = h(t) @ sigma by
    Magnus expansion to the first order.

        u(t+dt, dt) ≈ exp(-i * H(t+dt/2) * dt)
                    = cos(theta) * I - i sin(theta) * (h(t+dt/2) @ sigma) / |h(t+dt/2)|

    where theta = dt * |h(t+dt/2)|

    """
    ht = h(t+dt/2, *args)
    _h_ = np.linalg.norm(ht)
    theta = dt * _h_
    return np.cos(theta)*s0 - 1j * np.sin(theta) * np.tensordot(ht, s, axes=1) / _h_


def u(t, dt, h, *args):
    ts = np.arange(0, t, dt)
    ut = s0
    for t in ts:
        ut = u_dt(t, dt, h, *args) @ ut
    return ut
