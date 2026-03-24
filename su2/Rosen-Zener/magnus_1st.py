import numpy as np
from scipy.integrate import cumulative_trapezoid

from su2.common.utils import su2_exp
from su2.common.adiabatic.adiabatic_picture import eta_by_integral_sw
from su2.common.adiabatic.integrator import evolve


class AdiabaticSolver:
    def __init__(self, g, d, *args, **kwargs):
        self.g = g
        self.d = d
        self.args = args
        self.kwargs = kwargs

    def A_func(self, t):
        g = self.g
        return g / (2 * np.cosh(t))

    def E_func(self, t):
        g = self.g
        return -g * np.tanh(t) / (2 * np.cosh(t))

    def omega(self, t):
        """ instant energy"""
        return np.sqrt((self.d/2) ** 2 + (self.A_func(t)) ** 2)

    def wkb_psi_tp(self, ts):
        # TODO: optimize ts used to to integration

        omega_vals = self.omega(ts)

        # calculate phi(t) = 2 * ∫^t omega(s) ds
        phi_vals = cumulative_trapezoid(omega_vals, ts, initial=0.0)
        phi_vals = np.array(phi_vals)
        # fix gauge by setting φ(0) = 0
        phi0 = np.interp(0, ts, phi_vals)
        phi_vals -= phi0

        theta_d_vals = self.E_func(ts) * self.d / 2 / (omega_vals ** 2)
        eta_vals = - 1j * theta_d_vals * np.exp(2j * phi_vals) / 2

        a1_vals = cumulative_trapezoid(eta_vals, ts, initial=0.0)

        alpha, beta = su2_exp(a1_vals)

        data = {'ts': ts,
                'eta': eta_vals,
                'alpha': alpha,
                'beta': beta}
        return data

    def numeric_psi_tp(self, ts):
        data_raw = evolve(ts, self.A_func, self.E_func, *self.args, **self.kwargs)

        alpha, beta = data_raw['psi_t'].T

        data = {'ts': ts,
                'alpha': alpha,
                'beta': beta}
        return data


# def f(t, g, d):
#     return g / (2 * np.cosh(t))
#
#
# def f_dot(t, g, d):
#     return -g * np.tanh(t) / (2 * np.cosh(t))
#
#
# def omega(t, g, d):
#     return np.sqrt(d ** 2 + f(t, g, d) ** 2)
#
#
# def chi_dot(t, g, d):
#     return f_dot(t, g, d) * d / (2 * omega(t, g, d) ** 2)


# def eta_by_integral(ts, g, d):
#     """
#     Compute η(t) = -i * [dθ̇(t) / 2] * exp(2 i φ(t))
#
#     where dθ̇(t) = E(t) / ω(t)^2
#
#     """
#
#     def _integrand(x):
#         return omega(x, g, d)
#
#     phi_vals = cumulative_trapezoid(_integrand(ts), ts, initial=0.0)
#     phi_vals = np.array(phi_vals)
#
#     # fix gauge by setting φ(0) = 0
#     phi0 = np.interp(0, ts, phi_vals)
#     phi_vals -= phi0
#
#     theta_d_vals = f_dot(ts, g, d) * d / (omega(ts, g, d) ** 2)
#     return - 1j * theta_d_vals * np.exp(2j * phi_vals) / 2


# def wkb_psi_tp(ts, g, d):
#
#     eta_vals = eta_by_integral(ts, g, d)
#
#     a1_vals = cumulative_trapezoid(eta_vals, ts, initial=0.0)
#
#     alpha, beta = su2_exp(a1_vals)
#     return alpha, beta


def demo():
    """
    Demonstration of the first-order Magnus expansion for the Rosen-Zener model.
    Compares the numerical solution to the exact transition probability.
    """
    import matplotlib.pyplot as plt

    g = 1.0
    d = 3.0

    solver = AdiabaticSolver(g, d)

    t_vals = np.linspace(-10, 10, 1000)

    beta_magnus = solver.wkb_psi_tp(t_vals)['beta']
    beta_numerical = solver.numeric_psi_tp(t_vals)['beta']
    p_magnus = np.abs(beta_magnus) ** 2
    p_numeric = np.abs(beta_numerical) ** 2

    plt.figure(figsize=(8, 6))
    plt.plot(t_vals, p_magnus, label='Magnus Expansion β(t)')
    plt.plot(t_vals, p_numeric, label='Numerical β(t)', linestyle='--')

    plt.ylabel('df/dt')
    plt.title('Time derivative of the coupling function f(t)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    demo()
