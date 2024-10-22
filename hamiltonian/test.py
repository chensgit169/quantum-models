from hamiltonian.models.heisenberg import HeisenbergH
from hamiltonian.models.ising import IsingChain


def ising():
    ising = IsingChain(20)
    energy, psi = ising.get_ground_state()
    print(energy)
    print(psi)


def heisenberg():
    heisenberg = HeisenbergH(11)
    energy, psi = heisenberg.compute_eigenstates(k=20, which='SA')
    print(energy)
    # print(psi)


if __name__ == '__main__':
    heisenberg()
