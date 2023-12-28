from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from scipy.sparse.linalg import LinearOperator


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
