import numpy as np
from scipy.integrate import cumulative_trapezoid
from su2.common.pauli import s0, sx, sy

"""
Integrator of a two-level system under adiabatic representation,
especially used for Sauter-Schwinger effect. The Hamiltonian is

    H = [[1, A(t)+p],   -->     H = [[0, -if],
         [A(t)+p, -1]]               [if*, 0]]
        (original basis)          (adiabatic basis)
                
where f = E(t)/(2(1+A(t)^2)) * exp(2i phi(t)), and

        phi(t) = ∫^t sqrt(1 + A(s)^2) ds.
                    
The relation E(t) = -dA/dt needs to be implemented manually.


Chen 
First version: 2025-9-17
Last modified: 2025-11-27
"""


def eta_p(ts, A_func, E_func, p=0):
    A = np.asarray(A_func(ts)) + p
    E = np.asarray(E_func(ts))

    w = np.sqrt(1.0 + A ** 2)

    eta = (E / (2.0 * w ** 2))
    return eta


def cumulative_f_p(ts, A_func, E_func, p=0, fix_gauge=True):
    """
    Returns
    -------
    dict with keys:
        't'   : time array
        'w'   : w(t) = sqrt(1 + A(t)^2)
        'phi' : cumulative integral of w, ∫ w(s) ds
        'f'   : = E(t)/(2 w(t)^2) * exp(2 i phi(t))
    """
    ts = np.asarray(ts)
    if np.any(np.diff(ts) <= 0):
        raise ValueError("t must be strictly increasing")

    # find the t closest to zero to fix U(1) gauge

    # Step 1: compute A(t) and E(t)
    A = np.asarray(A_func(ts)) + p
    E = np.asarray(E_func(ts))

    # Step 2: compute w(t) and phi(t)
    w = np.sqrt(1.0 + A ** 2)
    phi = np.array(cumulative_trapezoid(w, ts, initial=0.0))  # phi = ∫ w

    if fix_gauge:
        phi_0 = np.interp(0.0, ts, phi)
        phi -= phi_0  # fix gauge by setting phi(t=0)=0

    # Step 3: compute f(t)
    f = (E / (2.0 * w ** 2)) * np.exp(2j * phi)

    return {'t': ts, 'w': w, 'phi': phi, 'f': f}


def safe_sin_div(f, dt, eps=1e-12):
    """ Compute sin(f*dt)/f safely for small f."""
    f_abs = np.abs(f)
    theta = f_abs * dt
    small = f_abs < eps
    res = np.ones_like(theta) * dt
    res[~small] = np.sin(theta[~small]) / f_abs[~small]
    return res


def evolve(ts, A_func, E_func, p=0, only_final=False):
    """
    Evolve the two-level system based on
        U(t+dt, t) ≈ exp(-i H(t) dt)
    and
        exp(-i theta n.sigma) = cos(theta) I - i sin(theta) n.sigma,
    which ensures unitarity.

    exp(-i [[0, -if], [if*, 0]] dt) =
        cos(|f| dt) I - i sin(|f| dt)/|f| * [[0, -if], [if*, 0]]

    Return the full time evolution data including psi(t) at all time steps.
    """
    dt = np.mean(np.diff(ts))

    data = cumulative_f_p(ts, A_func, E_func, p=p, fix_gauge=True)
    f_vals = data['f']
    theta_vals = np.abs(f_vals) * dt
    cos_vals = np.cos(theta_vals)
    ratio_vals = safe_sin_div(f_vals, dt)

    # initial state
    psi = np.array([1, 0])

    psi_t = []

    for cos_theta, ratio, f in zip(cos_vals, ratio_vals, f_vals):
        u = cos_theta * s0 - 1j * ratio * (np.real(f) * sy + np.imag(f) * sx)
        psi = u @ psi
        if only_final:
            continue
        psi_t.append(psi)

    # # TODO: vectorize this loop
    # for f in f_vals:
    #     f_abs = np.abs(f)
    #     theta = np.abs(f) * dt
    #     if f_abs < 1e-15:
    #         # for very small f
    #         ratio = dt
    #     else:
    #         ratio = np.sin(theta) / f_abs
    #
    #     u = np.cos(theta) * s0 - 1j * ratio * (np.real(f) * sy + np.imag(f) * sx)
    #     psi = u @ psi
    #     psi_t.append(psi)

    if only_final:
        return psi

    data['psi_t'] = np.array(psi_t)  # store the full time evolution
    return data
