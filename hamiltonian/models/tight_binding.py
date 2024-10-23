import numpy as np
from numpy import ndarray

from hamiltonian.base.base import Hamiltonian


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
        self._energy = None
        self._psi = None

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

    def get_whole_spectrum(self, **kwargs):
        """
        Avoid computing the whole spectrum each time.

        # TODO: use decorator to manage memory while avoiding unnecessary computation
        """
        if self._energy is None or self._psi is None:
            self._energy, self._psi = super().get_whole_spectrum(**kwargs)
        return self._energy, self._psi

    def psi_t(self, psi_0: ndarray, t: float, basis: str = "computation") -> ndarray:
        """
        Time evolution of the state vector psi.

        Args:
            psi_0: initial state vector
            t: time
            basis: basis for input, "computation" or "energy"
        """
        if not psi_0.shape == (self.m, ):
            raise ValueError("The length of psi_0 should be equal to site_number.")
        energy, psi = self.get_whole_spectrum()

        if basis == "computation":
            c_0 = (psi_0 @ psi)
        elif basis == "energy":
            c_0 = psi_0
        else:
            raise ValueError(f"basis should be either 'computation' or 'energy', not '{basis}'.")

        c_t = np.exp(-1j * energy * t) * c_0
        psi_t = psi @ c_t
        return psi_t


def test():
    n = 20
    e = 2
    u = 0

    model = TightBinding(n, e, u)
    energy, psi = model.get_whole_spectrum()
    assert np.allclose(energy, e * np.ones(n))


def time_evolve_plot():
    import matplotlib.pyplot as plt

    n = 20
    e = 2
    plt.figure()
    for u in [1, 2, 10]:

        model = TightBinding(n, e, u)

        psi_0 = np.zeros(n)
        n_i, n_f = 0, 2
        psi_0[n_i] = 1  # |psi_0> = |1>

        ts = np.linspace(0, 5, 5000)

        def calc_p(t: float):
            return np.abs(model.psi_t(psi_0, t)[n_f])**2

        p_n_f = np.array([calc_p(t) for t in ts])

        plt.plot(ts, p_n_f, label=f"$u={u:.1f}$")

    plt.xlabel("t")
    plt.title(f"Time evolution of $P_{n_f+1}(t)$, $\\epsilon=$const")
    plt.legend()

    # plt.show()
    plt.savefig("tight_binding_time_evolve_us.png", dpi=400)


if __name__ == '__main__':
    test()
    time_evolve_plot()
