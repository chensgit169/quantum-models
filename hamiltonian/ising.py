import numpy as np
from numpy import ndarray
from numpy.linalg import norm

from hamiltonian.base import Hamiltonian
from hamiltonian.operators import apply_ising_hamiltonian, apply_longitudinal_field, apply_transverse_field, apply_sigma


class IsingChain(Hamiltonian):
    def __init__(self, site_number: int, j: float = -1, use_pbc: bool = True):
        """
        Args:
            site_number:
            j: interaction constant, anti-ferromagnetic by default
            use_pbc: whether to use periodic boundary condition
        """
        Hamiltonian.__init__(self, site_number, 2, j, use_pbc)
        self.j = j
        self.use_pbc = use_pbc

    def _matvec(self, psi: np.ndarray) -> np.ndarray:
        """
        default: anti-ferromagnetic
        """
        # boundary term
        if self.use_pbc:
            _ = apply_sigma(psi, 3, self.n - 1)
            psi_out = apply_sigma(_, 3, 0)
        else:
            psi_out = np.zeros_like(psi)

        # inner term
        for i_psi in range(self.n - 1):
            _ = apply_sigma(psi, 3, i_psi)
            psi_out += apply_sigma(_, 3, i_psi + 1)
        return psi_out


class TransverseIsingChain(Hamiltonian):
    def __init__(self, site_number: int, h: float, g: float = 0.0, use_pbc: bool = True):
        Hamiltonian.__init__(self, site_number, 2, h, g, use_pbc)
        self.h = h
        self.g = g
        self.use_pbc = use_pbc

        # def longitudinal_matvec(psi_in: ndarray):
        #     return apply_longitudinal_field(psi_in, n=site_number)
        #
        # self.observable = LinearOperator((2 ** site_number, 2 ** site_number), matvec=longitudinal_matvec)

    def _matvec(self, psi: ndarray):
        psi_out = -(apply_ising_hamiltonian(psi, self.n, self.use_pbc)
                    + self.h * apply_transverse_field(psi, self.n)
                    + self.g * apply_longitudinal_field(psi, self.n))
        return psi_out

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'{self.n}, {self.h}, {self.g})')

    def __str__(self):
        return f'{self.n} sites Ising chain with transverse h={self.h}, longitudinal g={self.g}'

    def get_whole_spectrum(self):
        print(f'whole spectrum generated, N={self.n}, h={self.h}, g={self.g}')
        return super().get_whole_spectrum()

    def get_energy_level(self, k: int = None, which: str = None, with_indices: bool = True):
        """Default: total energy levels"""
        if k is None and which is None:
            energy, psi = self.get_whole_spectrum()
            indices = np.array(range(2 ** self.n))
        else:
            energy, psi = self.compute_eigenstates(k=k, which=which)
            indices = np.arange(k)
        if with_indices:
            return indices, energy
        else:
            return energy

    # Part 2, self-check function
    def eig_check(self, eig_val: float, eig_vec: ndarray):
        return np.abs(eig_val - eig_vec @ self.matvec(eig_vec))

    def spectrum_check(self, energy: ndarray, psi: ndarray, error_tolerance: float = 5e-10):
        assert psi.shape == (2 ** self.n, 2 ** self.n)
        eigen = True
        orthogonality = True
        normality = True
        ordered = np.all(energy == np.sort(energy))

        for i in range(2 ** self.n):
            e_error = self.eig_check(energy[i], psi[:, i])
            if e_error > error_tolerance:
                print(f'state {i}, energy={energy[i]}, eigen-value error = {e_error}')
                eigen = False

            n_error = abs(1 - norm(psi[:, i]))
            if n_error > error_tolerance:
                print(f'state {i}, energy={energy[i]}, normality error = {n_error}')
                normality = False

            for j in range(i + 1, 2 ** self.n):
                o_error = abs(psi[:, i] @ psi[:, j])
                if o_error > error_tolerance:
                    print(f'state ({i, j}), orthogonality error = {o_error}')
                    orthogonality = False
        print(f'Check under error tolerance {error_tolerance}:')
        print('eigen-check' + (1 - eigen) * ' NOT' + ' pass')
        print('normality-check' + (1 - normality) * ' NOT' + ' pass')
        print('orthogonality-check' + (1 - orthogonality) * ' NOT' + ' pass')
        print('energy level' + (1 - ordered) * ' NOT' + ' ordered')
        return None


def self_check(n_sites: int = 9, error_tolerance: float = 1e-11):
    rand_h = 0.5 + np.random.rand()
    rand_g = 0.5 + np.random.rand()

    numerical_ising = TransverseIsingChain(site_number=n_sites, h=rand_h, g=rand_g)
    es, ps = numerical_ising.get_whole_spectrum()
    numerical_ising.spectrum_check(es, ps, error_tolerance)
    return None


def analytical_verification(n_sites: int = 10, error_tolerance: float = 2e-13):
    from analytical.ising_analytical.analytical_ising_chain import AnalyticalIsingChain
    rand_h = 0.5 + np.random.rand()

    analytical_ising = AnalyticalIsingChain(h=rand_h, site_number=n_sites)
    numerical_ising = TransverseIsingChain(site_number=n_sites, h=rand_h)

    error = np.max(np.abs(
        analytical_ising.get_energy_level(with_indices=False) - numerical_ising.get_energy_level(with_indices=False)))
    assert error < error_tolerance, f'N={n_sites}, h={rand_h: .3f}, max energy error={error}'
    print(f'verified by theoretical TFIC, N={n_sites}, h={rand_h: .3f}: pass under error tolerance {error_tolerance}')
    return None
