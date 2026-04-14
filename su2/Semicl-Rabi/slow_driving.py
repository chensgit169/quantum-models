import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.special import ellipeinc
from tqdm import tqdm

from exact_solution import quasi_energy
from su2.common.utils import sinx_over_x
from su2.magnus import magnus_su2

plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 21


def load_params(which: int = 1):
    param_sets = yaml.safe_load(open('data/params.yaml'))['slow_driving']
    params = param_sets['params-set'+str(which)]
    d_min, d_max = params['d']['min'], params['d']['max']
    d_vals = np.linspace(d_min, d_max, 400)

    return d_vals


def g_func(t, a, d):
    m = np.sqrt(a ** 2 + d ** 2)
    k = a / m
    return ellipeinc(t, k ** 2)


def phi(t, a, d):
    m = np.sqrt(a ** 2 + d ** 2)
    return m * g_func(t, a, d)


def v_a(t, g, d):
    m = np.sqrt(g ** 2 + d ** 2)
    k = g / m

    cos = np.sin(t)
    gamma = k * np.sqrt(1 - k ** 2) * cos / (1 - k ** 2 * cos ** 2) / 2
    exp_i_phi = np.exp(1j * m * g_func(t, g, d))
    return gamma * exp_i_phi


def magnus_symmetric():
    d_vals = load_params()

    e_vals = np.array([quasi_energy(d, d) for d in tqdm(d_vals)])

    def eps_a(a_m, c_m):
        theta_0 = phi(np.pi, d_vals, d_vals) / 2
        theta_m = np.sqrt(np.abs(a_m) ** 2 + np.abs(c_m) ** 2)
        sin_eps_pi = np.sin(theta_0) * np.cos(theta_m) + c_m * np.cos(theta_0) * sinx_over_x(theta_m)
        eps = np.arcsin(sin_eps_pi.astype(complex)).real / np.pi
        return eps

    a1_m, c2_m, a3_m = np.array([magnus_su2(v_a, 0, np.pi, d, d) for d in d_vals]).T

    plt.figure(figsize=(8, 6))

    # 0-the approximation
    approx_0th = eps_a(0, 0)
    approx_1st = eps_a(a1_m, 0)
    approx_2nd = eps_a(a1_m, c2_m)
    approx_3rd = eps_a(a1_m + a3_m, c2_m)

    plt.plot(d_vals, approx_0th, '--', label='Adiabatic')
    # plt.plot(d_vals, approx_1st, '-.', label='1st Magnus')
    # plt.plot(d_vals, approx_2nd, ':', label='2nd Magnus')

    plt.plot(d_vals, approx_1st, ':', label='FMA')
    plt.plot(d_vals, approx_2nd, '-.', label='SMA')
    plt.plot(d_vals, approx_3rd, '--', label='TMA')

    line1 = plt.plot(d_vals, e_vals, label='Exact', color='k')
    color1 = line1[0].get_color()
    plt.plot(d_vals, -e_vals, color=color1)

    # plt.plot(d_vals, approx_3rd, ':', label='3rd Magnus')
    # plt.grid(True, alpha=0.3)
    plt.xlabel(r'$\Delta/\omega$')
    plt.ylabel(r'$\epsilon/\omega$')
    plt.legend(loc='best')
    plt.xlim(np.min(d_vals), np.max(d_vals))
    plt.tight_layout()
    # plt.savefig('figures/quasienergy/slow_driving/eps_magnus_symmetric_MA.pdf', dpi=400)
    plt.show()


if __name__ == '__main__':
    # adiabatic_limit()
    magnus_symmetric()
