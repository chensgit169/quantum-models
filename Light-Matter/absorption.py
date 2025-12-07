import numpy as np


def absorption_strength(psi, polarization, dipoles):
    """
    For given state of n-molecules, calculate the absorption strength from the state |G⟩ to the state |ψℓ⟩.

    Parameters:
    - psi (numpy array): Coefficients of the eigenstate |ψℓ⟩, a 1D array of length N.
    - polarization (numpy array): Polarization vector of the light-field, a 1D array (ϵx, ϵy, ϵz).
    - dipoles (numpy array): Dipole vectors of the molecules, a 2D array of size (N, 3), where each row is (μx, μy, μz).

    Returns:
    - A (float): The absorption strength A(ℓ) as defined in the equation.
    """
    # Calculate the dot product between the polarization vector and each dipole vector (vectorized)
    dipole_polarization = dipoles @ polarization

    # Calculate the absorption strength using vectorized operations
    A = np.abs(np.sum(psi * dipole_polarization)) ** 2

    return A
