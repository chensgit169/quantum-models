import numpy as np
from scipy.integrate import cumulative_trapezoid

from su2.common.adiabatic.adiabatic_picture import eta_by_integral, su2_exp
from su2.common.adiabatic.integrator import evolve

from sauter_pulse import sauter_pulse


class SauterSchwingerSolver:
    def __init__(self, A_func, E_func, m=1, *args, **kwargs):
        self.A_func = A_func
        self.E_func = E_func
        self.m = m
        self.args = args
        self.kwargs = kwargs

    def omega_p(self, t, p=0):
        """ instant energy"""
        return np.sqrt(self.m ** 2 + (p + self.A_func(t)) ** 2)

    def coupling_f(self, ts, p):
        return self.E_func(ts) / (2 * self.omega_p(ts, p) ** 2)

    def wkb_psi_tp(self, ts, p):
        # TODO: optimize ts used to to integration
        def omega(t): return self.omega_p(t, p)

        eta_vals = eta_by_integral(ts, self.E_func, omega)

        a1_vals = cumulative_trapezoid(eta_vals, ts, initial=0.0)

        alpha, beta = su2_exp(a1_vals)

        data = {'ts': ts,
                'p': p,
                'eta': eta_vals,
                'alpha': alpha,
                'beta': beta}
        return data

    def numeric_psi_tp(self, ts, p):
        data_raw = evolve(ts, self.A_func, self.E_func, p, *self.args, **self.kwargs)

        alpha, beta = data_raw['psi_t'].T

        data = {'ts': ts,
                'p': p,
                'alpha': alpha,
                'beta': beta}
        return data


class DoubleSauter(SauterSchwingerSolver):
    def __init__(self, e1, tau1, e2, tau2, t0=0.0, m=1.0, *args, **kwargs):
        self.e1, self.e2 = e1, e2
        self.tau1, self.tau2 = tau1, tau2
        self.t0 = t0

        A1, E1 = sauter_pulse(e1, tau1)
        A2, E2 = sauter_pulse(e2, tau2)

        def A_func(t): return A1(t) + A2(t - t0)

        def E_func(t): return E1(t) + E2(t - t0)

        super().__init__(A_func, E_func, m, *args, **kwargs)

    def psi_final(self, p, method='num', N=20001):
        t_lim = 10 * max(self.tau1, self.tau2, 1.0 / self.m)
        ts = np.linspace(-t_lim, t_lim, N)

        if method == 'num':
            psi = evolve(ts, self.A_func, self.E_func, p=p, only_final=True)
        elif method == 'wkb':
            data = self.wkb_psi_tp(ts, p)
            psi = (data['alpha'][-1], data['beta'][-1])
        else:
            raise ValueError(f"Unknown method: {method}")
        return psi


def demo():
    import matplotlib.pyplot as plt
    from sauter_pulse import sauter_pulse

    E0 = 1
    tau = 5.0
    A_func, E_func = sauter_pulse(E0, tau)

    sauter = SauterSchwingerSolver(A_func, E_func)

    t_min, t_max = -10, 10
    ts = np.linspace(t_min, t_max, 1000)
    p = 4

    fs = sauter.coupling_f(ts, p)

    # minimum of f
    min_index = np.argmin(fs)
    t_min_f = ts[min_index]

    tm = np.linspace(t_min_f, t_min_f + 20, 1000)
    tp = np.linspace(t_min_f - 20, t_min_f, 1000)
    fm = sauter.coupling_f(tm, p)
    fp = sauter.coupling_f(tp, p)

    # plt.plot(tm, fm, label='f(t) after min')
    # plt.plot(tp, fp, label='f(t) before min')
    plt.plot(tm, fp[::-1]-fm, label='')
    plt.show()
