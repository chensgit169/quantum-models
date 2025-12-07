import matplotlib.pyplot as plt
import numpy as np
from mpmath import hyp2f1, mp
from scipy.integrate import cumulative_trapezoid, quad, trapezoid
from tqdm import tqdm

from sauter_pulse import demo_p_dependence, sauter_pulse
from solver import SauterSchwingerSolver

plt.rcParams['font.size'] = 10
mp.dps = 30  # precision digits


class SauterSolution(SauterSchwingerSolver):
    def __init__(self, E0, tau, m=1):
        self.E0 = E0
        self.tau = tau
        A_func, E_func = sauter_pulse(E0, tau)
        super().__init__(self, A_func, E_func, m=m)

    def omega_0(self, p):
        """ limiting energy at t-> -infty"""
        return np.sqrt((p - self.E0 * self.tau) ** 2 + self.m ** 2)

    def omega_1(self, p):
        """ limiting energy at t-> +infty"""
        return np.sqrt((p + self.E0 * self.tau) ** 2 + self.m ** 2)

    def phi(self, ts, p, N=100000):
        assert np.all(np.diff(ts) > 0), "ts must be increasing"
        t_min, t_max = ts[0], ts[-1]

        ts_raw = np.linspace(t_min, t_max, N)
        omega_raw = self.omega_p(ts_raw, p)
        phi_raw = np.array(cumulative_trapezoid(omega_raw, ts_raw, initial=0))

        # interpolate to find values, reference phase at t=0
        phi_ref = np.interp(0, ts_raw, phi_raw)
        phi_vals = np.interp(ts, ts_raw, phi_raw) - phi_ref
        return phi_vals

    def phi0(self, p):
        # TODO: optimize the integral range

        def integrand(t):
            return - self.omega_p(t, p) + self.omega_0(p)

        result, error = quad(integrand, -30.0 * max(self.tau, 1 / self.m), 0, epsabs=1e-12)[:2]

        return result

    def phi1(self, p):
        def integrand(t):
            return self.omega_p(t, p) - self.omega_1(p)

        result, error = quad(integrand, 0, 30.0 * max(self.tau, 1 / self.m), epsabs=1e-12)[:2]
        return result

    def phi_ins(self, t, p, sign):
        """
        Instantaneous eigenstates
        """
        P = p + self.A_func(t)
        E = sign * self.omega_p(P)
        m = self.m

        norm = np.sqrt((E - P) ** 2 + m ** 2)
        psi1 = m / norm
        psi2 = (E - P) / norm
        return np.array([psi1, psi2])

    def psi_t(self, t, p):
        z = 1 / (1 + np.exp(-2 * t / self.tau))

        omega0 = self.omega_0(p)
        omega1 = self.omega_1(p)

        E0 = self.E0
        tau = self.tau
        m = self.m

        n0 = np.sqrt((omega0 - p + E0 * tau) ** 2 + m ** 2)
        a = -1j * omega0 * tau / 2
        b = 1j * omega1 * tau / 2
        c = 1j * E0 * tau ** 2

        _za_zb = (z ** a) * ((1 - z) ** b)

        psi1 = _za_zb * hyp2f1(a + b - c, a + b + c + 1, 2 * a + 1, z) * m / n0
        psi2 = _za_zb * hyp2f1(a + b + c, a + b - c + 1, 2 * a + 1, z) * (omega0 - p + E0 * tau) / n0
        return np.array([psi1, psi2])

    def abc(self, p):
        """
        point-wisely compute by mpmath
        """
        w0 = self.omega_0(p)
        w1 = self.omega_1(p)

        E0 = self.E0
        tau = self.tau

        p0 = p - E0 * tau
        p1 = p + E0 * tau

        a = -1j * w0 * tau / 2
        b = 1j * w1 * tau / 2
        c = 1j * E0 * tau ** 2

        A = mp.gamma(2 * a + 1) * mp.gamma(-2 * b) / (mp.gamma(a - b - c) * mp.gamma(a - b + c + 1))
        B = mp.gamma(2 * a + 1) * mp.gamma(2 * b) / (mp.gamma(a + b - c) * mp.gamma(a + b + c + 1))

        norm_0p = mp.sqrt((w0 + p0) / (2 * w0))
        norm_1p = mp.sqrt((w1 + p1) / (2 * w1))
        norm_1n = mp.sqrt((w1 - p1) / (2 * w1))

        phi0 = self.phi0(p)
        phi1 = self.phi1(p)

        alpha = A * norm_0p / norm_1p * mp.exp(1j * (-phi0 + phi1))
        beta = B * norm_0p / norm_1n * mp.exp(1j * (-phi0 - phi1))

        alpha = complex(alpha.real, alpha.imag)
        beta = complex(beta.real, beta.imag)

        return alpha, beta

    def rate(self, ps):
        """
        point-wisely compute by mpmath
        """
        from scipy.special import gamma

        w0 = self.omega_0(ps)
        w1 = self.omega_1(ps)

        E0 = self.E0
        tau = self.tau

        p0 = ps - E0 * tau
        p1 = ps + E0 * tau

        a = -1j * w0 * tau / 2
        b = 1j * w1 * tau / 2
        c = 1j * E0 * tau ** 2

        B = gamma(2 * a + 1) * gamma(2 * b) / (gamma(a + b - c) * gamma(a + b + c + 1))

        norm_0p = np.sqrt((w0 + p0) / (2 * w0))
        norm_1n = np.sqrt((w1 - p1) / (2 * w1))

        beta = B * norm_0p / norm_1n
        rate = 2 * np.abs(beta) ** 2

        return rate

    def total_pair_production_rate(self):
        """ total pair production rate per unit volume"""
        ps = np.linspace(0, 10 * max(self.E0 * self.tau, self.m), 1000)
        rates = self.rate(ps)
        assert min(rates) / max(rates) < 1e-4, "Integration range not large enough, increase upper limit"

        total_rate = 2 * trapezoid(rates, ps)
        return total_rate

    def classical_r(self, p):
        E0 = self.E0
        tau = self.tau

        # avoided crossing for p
        if np.abs(p) < E0 * tau:
            tp = tau * np.arctanh(p / (E0 * tau))
            ts = np.linspace(0, tp, 1000)
            ws = self.omega_p(ts, p)
            r = 2 * trapezoid(ws, ts)
        else:
            r = self.phi0(p)
        return r


def demo_phase(item='phase'):
    from params import e1, tau1

    tau = 20 * tau1
    e = 1 * e1

    A0 = e * tau
    print(f"A0={A0}, tau={tau}")

    sauter = SauterSolution(E0=e, tau=tau)

    p_vals = np.linspace(0.788, 0.789, 1000)

    alpha_vals, beta_vals = np.array([sauter.abc(p) for p in tqdm(p_vals)]).T

    plt.figure(figsize=(8, 4))
    phase = np.angle(beta_vals/alpha_vals)
    phase = np.mod(phase, 2 * np.pi)

    if item == 'phase':
        plt.plot(p_vals, phase / np.pi, label=r'Phase of $\beta/\alpha$')
    elif item == 'ab':
        plt.plot(p_vals, np.angle(alpha_vals), label=r'Phase of $\alpha$')
        plt.plot(p_vals, np.angle(beta_vals), label=r'Phase of $\beta$')
    elif item == 'diff':
        dphi_dp = np.gradient(phase, p_vals)
        # p_classical = np.array([sauter.classical_r(p) for p in tqdm(p_vals)])
        plt.plot(p_vals, dphi_dp, label=r'$d\phi/dp$')

        # r_classical = np.gradient(p_classical, p_vals)
        # plt.plot(p_vals, r_classical, label='Classical r')

        plt.legend()
        plt.yscale('log')
        # plt.ylim(0, 1.1 * np.max(dphi_dp))

    r_p0(e, tau)

    plt.xlim(np.min(p_vals), np.max(p_vals))
    plt.xlabel('p')
    plt.ylabel('phase')

    plt.legend()
    plt.grid(True)
    plt.show()


def demo():
    from params import e1, tau1

    sauter_solution = SauterSolution(E0=e1, tau=tau1)

    data_filename = 'data/single_sauter/' + f'exact_p_spectrum_E={e1:.2f}_tau={tau1:.2f}'

    ps = np.linspace(0, 5 * max(e1 * tau1, sauter_solution.m), 2000)

    alpha_vals, beta_vals = np.array([sauter_solution.abc(p) for p in tqdm(ps)]).T
    np.savez(data_filename,
             tau=tau1,
             e0=e1,
             ps=ps,
             alpha=alpha_vals,
             beta=beta_vals
             )

    # phase_vals = np.angle(beta_vals / alpha_vals)
    # phase_vals = np.mod(phase_vals, 2 * np.pi)

    demo_p_dependence(ps, alpha_vals, beta_vals, item='rate')

    plt.grid(True)
    # plt.yscale('log')
    plt.show()


def demo_phi01():
    from params import e1, tau1

    sauter_solution = SauterSolution(E0=e1, tau=tau1)

    print(sauter_solution.phi0(0) + sauter_solution.phi1(0))

    p_vals = np.linspace(-3 * e1 * tau1, 3 * e1 * tau1, 300)

    phi0_vals = np.array([sauter_solution.phi0(p) for p in tqdm(p_vals)])
    phi1_vals = np.array([sauter_solution.phi1(p) for p in tqdm(p_vals)])

    plt.figure(figsize=(8, 4))
    plt.plot(p_vals, phi0_vals, label=r'$\phi_0$')
    plt.plot(p_vals, phi1_vals, label=r'$\phi_1$')

    diff = phi0_vals + phi1_vals[::-1]
    plt.plot(p_vals, diff, label=r'$\phi_0+\phi_1$')

    plt.xlabel('p')
    plt.ylabel('phase')

    plt.legend()
    plt.grid(True)
    plt.show()


def tau_dependence():
    plt.figure(figsize=(8, 4))

    for e1 in [0.5, .75, 1.0, 1.25]:

        rate_vals = []
        tau_vals = np.linspace(1e-4, 3, 500)
        for tau in tqdm(tau_vals):
            sauter_solution = SauterSolution(E0=e1, tau=tau)
            total_rate = sauter_solution.total_pair_production_rate()
            rate_vals.append(total_rate)
        plt.plot(tau_vals, rate_vals, label=f'$E_0$={e1:.2f}$E_c$')

    plt.xlim(np.min(tau_vals), np.max(tau_vals))
    plt.xlabel(r'$\tau/m$')
    plt.ylabel('Total pair production rate')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig('figures/single_sauter/total_rate_vs_tau.pdf', dpi=400)
    plt.show()


def demo_rate_p():
    from params import e1

    tau = 1e-7
    sauter_solution = SauterSolution(E0=e1, tau=tau)
    ps = np.linspace(30, 40 * max(e1 * tau, 1), 2000)
    rates = sauter_solution.rate(ps)
    plt.figure(figsize=(8, 4))
    plt.plot(ps, rates)
    plt.xlim(np.min(ps), np.max(ps))
    plt.ylim(0, np.max(rates) * 1.1)
    # plt.yscale('log')
    plt.xlabel(r'$p$')
    plt.ylabel('Pair production rate')
    plt.show()


def r_p0(e, tau):
    from su2.common.math.gamma_phase import deriv_arg_gamma

    sauter_solution = SauterSolution(E0=e, tau=tau)
    print(tau)
    w0 = sauter_solution.omega_0(0)

    r_gamma = 2 * (e * tau ** 2 / w0 *
               (deriv_arg_gamma(w0 * tau) - deriv_arg_gamma(e * tau ** 2)))
    print("r_gamma =", r_gamma)

    # dp = 1e-5
    # phi = sauter_solution.phi0(p=0) + sauter_solution.phi1(p=0)
    # phi_plus = sauter_solution.phi0(p=dp) + sauter_solution.phi1(p=dp)
    # dphi_dp = (phi_plus - phi) / dp
    # print("dphi/dp at p=0:", dphi_dp)

    from scipy.integrate import quad

    def integrand(y):
        return np.tanh(y) / np.sqrt(1 ** 2 + (e * tau * np.tanh(y)) ** 2) - 1 / np.sqrt(1 ** 2 + (e * tau) ** 2)

    # 积分
    I, err = quad(integrand, 0, np.inf, limit=200)

    # 最终 dφ/dp
    dphi_dp = 2 * e * tau ** 2 * I

    print("dφ/dp at p=0 =", dphi_dp)

    r_tot = - dphi_dp + r_gamma
    print("r_tot at p=0 =", r_tot)

    print("2m/E=", 2/e)
    return r_tot


if __name__ == '__main__':
    # tau_dependence()
    # demo_rate_p()
    # demo_phase('phase')
    # demo_phi01()
    r_p0(e=0.5, tau=400)
