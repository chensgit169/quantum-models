import numpy as np
from numpy import ndarray
from numpy.linalg import norm

from hamiltonian.base import Hamiltonian
from hamiltonian.operators import apply_ising_hamiltonian, apply_longitudinal_field, apply_transverse_field, apply_sigma


class Hubbard(Hamiltonian):
    def __init__(self, state_number: int,
                 eps: ndarray,
                 u: float = 1, use_pbc: bool = False):
        """
        Args:
            state_number:
            u: interaction constant, anti-ferromagnetic by default
            use_pbc: whether to use periodic boundary condition
        """
        Hamiltonian.__init__(self, state_number, 1, u, use_pbc)
        self.j = u
        self.eps = eps
        if not len(eps) == state_number:
            raise ValueError("The length of eps should be equal to site_number.")
        self.use_pbc = use_pbc

    def _matvec(self, psi: np.ndarray) -> np.ndarray:
        """

        """
        psi_out = psi.copy()
        for i_psi in range(self.n):
            psi_out += self.eps[i_psi] * self.j * apply_sigma(psi, 0, i_psi)
        for i_psi in range(self.n-1):
            psi_out += self.j * apply_sigma(psi, 3, i_psi)

        return psi_out

    def apply_hopping(self, psi: np.ndarray, i: int, j: int):
        psi_out = psi.copy()
        psi_out += self.j * apply_sigma(psi, 1, i) * apply_sigma(psi, 0, j)
        psi_out += self.j * apply_sigma(psi, 0, i) * apply_sigma(psi, 1, j)
        return psi_out
