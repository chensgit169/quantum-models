import numpy as np
from numpy.linalg import eigh


def calculate_v_point_dip(mu1, mu2, R):
    """
    Calculate the point dipole interaction energy.
    CGS units are used, i.e., mu in esu*cm = 10^18 Debye and R in cm.

    Parameters:
    - mu1: numpy array, the dipole moment vector of the first dipole (e.g., [μ1x, μ1y, μ1z]).
    - mu2: numpy array, the dipole moment vector of the second dipole (e.g., [μ2x, μ2y, μ2z]).
    - R: numpy array, the vector between the two dipoles (e.g., [Rx, Ry, Rz]).

    Returns:
    - VPointDip: float, the calculated interaction energy.
    """
    R_magnitude = np.linalg.norm(R)  # Calculate the magnitude of R
    R_unit = R / R_magnitude  # Calculate the unit vector of R

    mu1_dot_mu2 = np.dot(mu1, mu2)  # Dot product of μ1 and μ2
    mu1_dot_R = np.dot(mu1, R_unit)  # Dot product of μ1 and unit vector of R
    mu2_dot_R = np.dot(mu2, R_unit)  # Dot product of μ2 and unit vector of R

    VPointDip = (1 / R_magnitude**3) * (mu1_dot_mu2 - 3 * mu1_dot_R * mu2_dot_R)

    return VPointDip


def calculate_interaction_matrix(mu, R):
    """
    Calculate the dipole-dipole interaction matrix for a set of dipoles.

    Parameters:
    - mu: numpy array (N×3), the dipole moment vectors for N dipoles.
    - R: numpy array (N×3), the position vectors of the N dipoles.
    - epsilon: float, the permittivity of the medium.

    Returns:
    - V: numpy array (N×N), the interaction matrix where V[i][j] is the interaction
         energy between dipole i and dipole j.
    """
    N = len(mu)  # Number of dipoles
    v = np.zeros((N, N))  # Initialize the interaction matrix

    for i in range(N):
        for j in range(N):
            if i == j:
                # No self-interaction
                v[i, j] = 0
            else:
                # Compute R_ij
                Rij = R[j] - R[i]
                v[i, j] = calculate_v_point_dip(mu[i], mu[j], Rij)
    return v


def participation_ratio(psi):
    """
    Calculate the participation ratio D for a quantum state.

    Parameters:
    - psi: numpy array, coefficients of the state |ψ⟩ in the basis |πn⟩.

    Returns:
    - D: float, the participation ratio.
    """
    pn = np.abs(psi) ** 2
    D = np.sum(1 / pn)
    return D


def demo_point_dip():
    mu1 = np.array([1.0, 0.0, 0.0])  # Example dipole moment vector for μ1
    mu2 = np.array([1.0, 1.0, 0.0])  # Example dipole moment vector for μ2
    R = np.array([0.0, 0.0, 1.0])    # Distance vector between the dipoles

    VPointDip = calculate_v_point_dip(mu1, mu2, R)
    print(f"VPointDip = {VPointDip} erg")


def demo_v_mat():
    # Example usage
    mu = np.array([
        [1.0, 0.0, 0.0],  # Dipole moment for dipole 1
        [1.0, 1.0, 0.0],  # Dipole moment for dipole 2
        [0.0, 1.0, 1.0],  # Dipole moment for dipole 3
    ])

    R = np.array([
        [0.0, 0.0, 0.0],  # Position of dipole 1
        [1.0, 0.0, 0.0],  # Position of dipole 2
        [0.0, 1.0, 0.0],  # Position of dipole 3
    ])

    v = calculate_interaction_matrix(mu, R)

    print("Interaction matrix V:")
    print(v)

    es, psis = eigh(v)
    for i, e in enumerate(es):
        print(f"Eigenvalue {i}: {e:.4f}", end=' ')
        print(f"Eigenvector {i}: {psis[:, i]}", end=' ')
        print(f"Participation ratio: {participation_ratio(psis[:, i])}")


if __name__ == '__main__':
    demo_v_mat()
