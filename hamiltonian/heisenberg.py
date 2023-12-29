import numpy as np

from hamiltonian.base import Hamiltonian
from hamiltonian.operators import apply_sigma


class HeisenbergH(Hamiltonian):
    def __init__(self, site_number: int, j: float = -1, h: float = 0, use_pbc: bool = True):
        """
        Args:
            site_number:
            j: interaction constant, anti-ferromagnetic by default
            h: external magnetic field
            use_pbc: whether to use periodic boundary condition
        """
        Hamiltonian.__init__(self, site_number, 2, j, h, use_pbc, dtype=complex)
        self.j = j
        self.h = h
        self.use_pbc = use_pbc

    def _matvec(self, psi: np.ndarray) -> np.ndarray:
        def apply_ss(i_s: int):
            """
            Spin-1/2 SS interaction
            """
            # boundary term
            if self.use_pbc:
                _ = apply_sigma(psi, i_s, self.n - 1)
                psi_out = apply_sigma(_, i_s, 0)
            else:
                psi_out = np.zeros_like(psi)

            # inner term
            for i_psi in range(self.n - 1):
                _ = apply_sigma(psi, i_s, i_psi)
                psi_out += apply_sigma(_, i_s, i_psi + 1)
            return psi_out

        return apply_ss(1) + apply_ss(2) + apply_ss(3)
