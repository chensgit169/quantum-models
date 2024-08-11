import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


class TwoLevelInField:
    def __init__(self, omega, epsilon, gap=1.0):
        """
        omega: float, the frequency of the driving field
        epsilon: float, the amplitude of the driving field
        gap: float, the energy gap E between the two states, E=1 as the energy unit
        """
        self.omega = omega
        self.epsilon = epsilon
        self.gap = gap

    def hamiltonian(self, t):
        raise NotImplementedError

    def __call__(self, t, psi, *args, **kwargs):
        return -1j * self.hamiltonian(t) @ psi

    @property
    def delta(self):
        """
        omega_0 - E
        """
        return self.omega - self.gap


class FullHamiltonian(TwoLevelInField):
    def hamiltonian(self, t):
        return np.array([[-1/2, self.epsilon * np.cos(self.omega * t)],
                         [self.epsilon * np.cos(self.omega * t), 1/2]])


class RWAHamiltonian(TwoLevelInField):

    @property
    def rabi_freq(self):
        return np.sqrt(self.epsilon ** 2 + self.delta ** 2)

    @property
    def a_p(self):
        return self.epsilon / np.sqrt(self.epsilon ** 2 + (self.rabi_freq+self.delta) ** 2)

    @property
    def a_m(self):
        return self.epsilon / np.sqrt(self.epsilon ** 2 + (self.rabi_freq-self.delta) ** 2)

    def psi_p(self, t):
        return (np.exp(-1j*self.rabi_freq*t/2) *
                np.array([self.a_p * np.exp(+1j * self.rabi_freq * t / 2),
                          -self.a_m * np.exp(-1j * self.rabi_freq * t / 2)]))

    def psi_m(self, t):
        return (np.exp(+1j*self.rabi_freq*t/2) *
                np.array([self.a_m * np.exp(+1j * self.rabi_freq * t / 2),
                          self.a_p * np.exp(-1j * self.rabi_freq * t / 2)]))

    def psi(self, t, cp, cm):
        """
        Exact solution of the Schrodinger equation
        cp, cm: complex, the coefficients of the two states
        """
        return cp * self.psi_p(t) + cm * self.psi_m(t)

    def hamiltonian(self, t):
        return np.array([[-1, self.epsilon * np.exp(1j * self.omega * t)],
                         [self.epsilon * np.exp(-1j * self.omega * t), 1]]) / 2

    def __call__(self, t, psi, *args, **kwargs):
        return -1j * self.hamiltonian(t) @ psi


def main():
    omega = 6.0
    epsilon = 2.0

    h_rwa = RWAHamiltonian(omega, epsilon)
    h_full = FullHamiltonian(omega, epsilon)

    t_span = (0, 6 * np.pi / h_rwa.rabi_freq)
    t_eval = np.linspace(*t_span, 1000)
    psi0 = np.array([1.0, 0.0], dtype=complex)

    solu_rwa = solve_ivp(h_rwa, t_span, psi0, t_eval=t_eval, method='RK45')
    solu_full = solve_ivp(h_full, t_span, psi0, t_eval=t_eval, method='RK45')

    p1_rwa_num = np.abs(solu_rwa.y[0]) ** 2
    p1_rwa_ana = np.abs(h_rwa.psi(solu_rwa.t, h_rwa.a_p, h_rwa.a_m)[0]) ** 2
    p1_full = np.abs(solu_full.y[0]) ** 2

    plt.plot(solu_rwa.t, p1_rwa_ana, '-.', label=r'RAW(Analytic)')
    plt.plot(solu_rwa.t, p1_rwa_num, label=r'RWA(Numeric)')
    plt.plot(solu_full.t, p1_full, '--', label=r'Full Hamiltonian')

    plt.title("$"+f"\\Delta={omega-1:.2f}, \\epsilon={epsilon:.2f}"+"$")
    plt.xlabel('Time')
    plt.ylabel('$P_1$')
    plt.legend()
    plt.savefig("off_resonance_strong_field.png", dpi=400)


if __name__ == '__main__':
    main()