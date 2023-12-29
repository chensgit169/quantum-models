import numpy as np

from .base import Hamiltonian
from .operators import apply_ss


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
        self.coupling_pairs = [(i, i+1) for i in range(self.n - 1)]
        if self.use_pbc:
            self.coupling_pairs.append((self.n - 1, 0))

    def _matvec(self, psi: np.ndarray) -> np.ndarray:
        def _ss(i_s: int):
            """
            Spin-1/2 SS interaction
            """
            psi_out = np.sum([apply_ss(psi, i_s, i1, i2) for i1, i2 in self.coupling_pairs], axis=0)
            return psi_out

        return _ss(1) + _ss(2) + _ss(3)
