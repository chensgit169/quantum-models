import numpy as np
from numpy import ndarray

from hamiltonian.base import Hamiltonian


class TightBinding(Hamiltonian):
    def __init__(self, state_number: int,
                 eps: ndarray | float | int,
                 u: float = 1.,
                 use_pbc: bool = False):
        """
        Args:
            state_number:
            eps: on-site energy, optionally varying with site
            u: hopping constant, use as unit by default
            use_pbc: whether to use periodic boundary condition
        """
        Hamiltonian.__init__(self, 1, state_number, use_pbc)
        self.u = u
        self.eps = eps

        if isinstance(eps, (float, int)):
            self.eps = np.array([eps] * state_number)
        elif not len(eps) == state_number:
            raise ValueError("The length of eps should be equal to site_number.")
        self.use_pbc = use_pbc

    def _matvec(self, psi: np.ndarray) -> np.ndarray:
        """
        Energy band + nearst hopping term
        """
        if not len(psi) == self.m:
            raise ValueError("The length of psi should be equal to site_number.")

        psi_out = psi * self.eps
        psi_out[1:] += self.u * psi[:-1]
        psi_out[:-1] += self.u * psi[1:]

        if self.use_pbc:
            psi_out[0] += self.u * psi[-1]
            psi_out[-1] += self.u * psi[0]

        return psi_out


def test():
    n = 20
    e = 2
    u = 0

    model = TightBinding(n, e, u)
    energy, psi = model.get_whole_spectrum()
    assert np.allclose(energy, e * np.ones(n))


if __name__ == '__main__':
    test()
