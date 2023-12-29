from numpy import ndarray
import numpy as np

"""
Commonly used operators and interaction terms in quantum mechanics.
"""


# Part 1, local operators
def apply_sigma(psi: ndarray, i_s: int, i_psi: int):
    """
    Apply Pauli sigma^i_s to the i_psi-th qubit of psi.
    """
    sx = np.array([[0, 1], [1, 0]])
    sy = np.array([[0, -1j], [1j, 0]])
    sz = np.array([[1, 0], [0, -1]])
    s0 = np.array([[1, 0], [0, 1]])
    sigma = {1: sx, 2: sy, 3: sz, 0: s0}[i_s]

    psi = psi.reshape((2**i_psi, 2, -1))
    psi_out = np.tensordot(sigma, psi, axes=[[1], [1]]).transpose((1, 0, 2)).reshape(-1)
    return psi_out


# Interaction terms
def apply_ss(psi, i_s: int, i1, i2):
    """
    Spin-1/2 SS interaction
    """
    assert i1 != i2
    _ = apply_sigma(psi, i_s, i1)
    psi_out = apply_sigma(_, i_s, i2)
    return psi_out


# Part 2, global operators, with n always explicitly specified
def apply_parity_operator(psi: ndarray, n: int):
    """ continued multiplication of sigma^z_i """
    for i in range(n):
        psi = apply_sigma(psi, 3, i)
    return psi


def apply_ising_hamiltonian(psi: ndarray, n: int, use_pbc: bool = True):
    """
    default: anti-ferromagnetic
    """
    # boundary term
    if use_pbc:
        _ = apply_sigma(psi, 3, n-1)
        psi_out = apply_sigma(_, 3, 0)
    else:
        psi_out = np.zeros_like(psi)

    # inner term
    for i_psi in range(n-1):
        _ = apply_sigma(psi, 3, i_psi)
        psi_out += apply_sigma(_, 3, i_psi+1)
    return psi_out


def apply_transverse_field(psi: ndarray, n: int):
    psi_out = np.sum([apply_sigma(psi, 1, i_psi) for i_psi in range(n)], axis=0)
    return psi_out


def apply_longitudinal_field(psi: ndarray, n: int):
    psi_out = np.sum([apply_sigma(psi, 3, i_psi) for i_psi in range(n)], axis=0)
    return psi_out


def apply_staggered_longitudinal_field(psi: ndarray, n: int):
    psi_out = np.sum([(-1)**i_psi * apply_sigma(psi, 3, i_psi) for i_psi in range(n)], axis=0)
    return psi_out


# Part 3, correlation measurements
def measure_sigma_zi(psi0: ndarray, psi1: ndarray, i_psi: int):
    psi0.reshape(2**i_psi, 2, -1)[:, 1, :] *= -1
    return np.sum(psi1 @ psi0)





