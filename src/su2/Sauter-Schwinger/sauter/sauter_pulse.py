import numpy as np
import matplotlib.pyplot as plt
from su2.common.adiabatic.integrator import eta_p


def sech2(x, threshold=350):
    """
    Compute sech^2(x) = 1 / cosh^2(x) and avoid overflow due to large x.
    """
    large_mask = np.abs(x) > threshold
    safe_x = x[~large_mask]

    result = np.zeros_like(x)  # for large x, sech^2(x) ~ 0
    result[~large_mask] = 1.0 / np.cosh(safe_x) ** 2
    return result


def sauter_pulse(E0, tau):
    """
    return A(t) and E(t) for a Sauter pulse as Callable functions.
    """

    def A(t): return E0 * tau * np.tanh(t / tau)

    def E(t): return - E0 * sech2(t / tau)

    return A, E


def load_double_sauter(param_set):
    tau1 = param_set['tau1']
    e1 = param_set['e1']
    tau2 = param_set['tau2']
    e2 = param_set['e2']
    t0 = param_set['t0']

    A1, E1 = sauter_pulse(e1, tau1)
    A2, E2 = sauter_pulse(e2, tau2)
    def A_func(t): return A1(t) + A2(t - t0)
    def E_func(t): return E1(t) + E2(t - t0)
    return A_func, E_func


def demo_p_dependence(ps, alpha_vals, beta_vals, item):
    rates = 2 * np.abs(beta_vals) ** 2

    plt.figure(figsize=(8, 4))
    plt.xlabel('p [m]')

    if item == 'rate':
        plt.plot(ps, rates, label=r'Pair production rate $2|\beta|^2$')
        plt.ylabel(r'$2|\beta|^2$')
    elif item == 'phase':

        phases = np.angle(beta_vals / alpha_vals)
        phases = np.mod(phases, 2 * np.pi)

        plt.plot(ps, phases / np.pi)
        plt.ylabel('Phase of beta/alpha [rad]')
    elif item == 'phase_diff':
        phases = np.angle(beta_vals / alpha_vals)
        phases = np.unwrap(phases, 1.7 * np.pi)

        phase_diff = np.gradient(phases, ps)

        plt.plot(ps, phase_diff / np.pi)
        plt.ylabel('d(Phase of beta/alpha)/dp [rad/m]')

    elif item == 'alpha_im':
        plt.plot(ps, alpha_vals.imag, label='Im(alpha)')
        plt.ylabel('Im(alpha)')
    elif item == 'beta':
        line_b = plt.plot(ps, beta_vals.real, label='beta')
        color_b = line_b[0].get_color()
        plt.plot(ps, beta_vals.imag, '--', color=color_b)

    plt.grid(True)


def demo_eta():
    p = 0

    tau1 = 50
    tau2 = tau1 * 1e-2
    t0 = 0 * tau1 / 4  # shift
    e1 = 0.25  # field strength, in unit of E_c
    e2 = 1 * 0.025

    A1, E1 = sauter_pulse(e1, tau1)
    A2, E2 = sauter_pulse(e2, tau2)

    def A_func(t): return A1(t) + A2(t - t0)

    def E_func(t): return E1(t) + E2(t - t0)

    t_vals = np.linspace(-0.2 * tau1, 0.2 * tau1, 3000)

    eta_vals_single = eta_p(t_vals, A1, E1, p)
    eta_vals_double = eta_p(t_vals, A_func, E_func, p)

    plt.figure(figsize=(8, 4))

    # show eta(t)
    # plt.plot(t_vals, eta_vals_double)
    # plt.plot(t_vals, eta_vals_single, '--')
    # plt.ylabel('eta(t)')

    # show eta'(t)
    d_eta_vals_double = np.gradient(eta_vals_double, t_vals)
    d_eta_vals_single = np.gradient(eta_vals_single, t_vals)
    plt.plot(t_vals, d_eta_vals_double, label='Double Sauter pulse')
    plt.plot(t_vals, d_eta_vals_single, '--', label='Single Sauter pulse')
    plt.ylabel("d eta(t)/dt")

    plt.legend()

    plt.xlabel('t [1/m]')

    plt.title('Adiabatic parameter eta(t)')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    demo_eta()