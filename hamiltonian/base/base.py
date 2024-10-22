from abc import abstractmethod
from typing import Any

import numpy as np

from scipy.sparse.linalg import LinearOperator, eigsh


class Hamiltonian(LinearOperator):
    """
    Base class for Hamiltonian of quantum models.
    """

    def __init__(self, n: int, m: int, *args, dtype: Any = float):
        """
        Args:
            n: site number
            m: dimension of local Hilbert space
            args: interaction constants
            dtype: float or complex
        """
        self.n = n
        self.m = m
        self.args = args
        LinearOperator.__init__(self, shape=(m ** n, m ** n), dtype=dtype)

    @abstractmethod
    def _matvec(self, psi: np.ndarray) -> np.ndarray:
        """
        Apply the Hamiltonian to the wave function.

        Args:
            psi: state vector

        Returns:
            state: state vector
        """
        raise NotImplementedError

    # Part 1, eigen-states computation
    def compute_eigenstates(self, k: int, which: str = 'SA', v0=None, **kwargs):
        """"""
        energy, psi = eigsh(self, k=k, which=which, v0=v0, tol=1e-10, **kwargs)
        return energy, psi

    def get_ground_state(self, v0=None):
        energy, psi = self.compute_eigenstates(k=1, which='SA', v0=v0, tol=None)
        return energy[0], psi[:, 0]

    def get_whole_spectrum(self, **kwargs):
        """
        Since eigsh does not support computing all eigenvalues, we split.

        TODO: set limitation on the number of eigenvalues
        """
        k1 = self.m ** self.n - 1
        k2 = 1
        energy_small, psi_small = self.compute_eigenstates(k=k1, which='SA', **kwargs)
        energy_large, psi_large = self.compute_eigenstates(k=k2, which='LA', **kwargs)

        energy = np.concatenate((energy_small, energy_large))
        psi = np.concatenate((psi_small, psi_large), axis=1)
        return energy, psi

    # Part 2, pure state and thermal expectation
    def expectation(self, psi: np.ndarray):
        return psi.copy() @ self.matvec(psi)

    def partition(self, beta: float):
        # TODO
        raise NotImplementedError
