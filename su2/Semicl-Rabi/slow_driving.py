import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipeinc
from tqdm import tqdm

from exact_solution import quasi_energy
import yaml

from su2.common.magnus.magnus_su2 import a3_integral, a1_integral, c2_integral


plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 21


def sinx_over_x(x):
    """Compute sin(x)/x safely for x near 0."""
    res = np.ones_like(x)
    mask = np.abs(x) > 1e-8
    res[mask] = np.sin(x[mask]) / x[mask]
    return res


def g_func(t, a, d):
    m = np.sqrt(a ** 2 + d ** 2)
    k = a / m
    return ellipeinc(t, k ** 2)


def phi(t, a, d):
    m = np.sqrt(a ** 2 + d ** 2)
    return m * g_func(t, a, d)


def v(t, a, d):
    m = np.sqrt(a ** 2 + d ** 2)
    k = a / m

    cos = np.sin(t)
    gamma = k * np.sqrt(1 - k ** 2) * cos / (1 - k ** 2 * cos ** 2) / 2
    exp_i_phi = np.exp(1j * m * g_func(t, a, d))
    return gamma * exp_i_phi


def insert_nans_at_jumps(x, y, jump_size=0.5):
    """
    Insert NaN values into x and y arrays at points where a large jump occurs,
    using vectorized NumPy operations (no Python loops).
    """
    dy = np.abs(np.diff(y))

    # Indices where jump occurs (i.e., |Δy| > threshold)
    jump_indices = np.where(dy > jump_size)[0]

    if jump_indices.size == 0:  # No jumps found
        return x, y

    # Compute insertion positions (after each jump index)
    insert_positions = jump_indices + 1

    # Insert NaNs into x and y arrays at the detected positions
    x_out = np.insert(x, insert_positions, np.nan)
    y_out = np.insert(y, insert_positions, np.nan)
    return x_out, y_out


def adiabatic_limit():
    param_sets = yaml.safe_load(open('data/params.yaml'))
    params = param_sets['slow_driving']['params-set1']
    d_min, d_max = params['d']['min'], params['d']['max']
    d_vals = np.linspace(d_min, d_max, 400)

    eps_exact_vals = np.array([quasi_energy(d, d, real_only=True) for d in d_vals])

    phi_vals = phi(np.pi, d_vals, d_vals) / (2 * np.pi)  # at T/2
    eps_adiabatic_vals = (phi_vals + 1 / 2) % 1 - 1 / 2

    line1 = plt.plot(d_vals, eps_exact_vals, label='Exact')
    color1 = line1[0].get_color()
    plt.plot(d_vals, -eps_exact_vals, color=color1)

    # plt.title(r'Quasienergy along $\Delta=A$')
    plt.xlabel(r'$\Delta=A$')
    plt.ylabel(r'$\epsilon$')
    # plt.grid()

    plt.legend(loc='lower right')
    # plt.xlim(10.67, 10.70)
    # plt.ylim(0.4975, 0.5)
    plt.tight_layout()
    plt.savefig('figures/quasienergy/slow_driving/eps_adiabatic_limit.png', dpi=400)
    plt.show()


def magnus_symmetric():
    param_sets = yaml.safe_load(open('data/params.yaml'))
    params = param_sets['slow_driving']['params-set1']
    d_min, d_max = params['d']['min'], params['d']['max']
    d_vals = np.linspace(d_min, d_max, 400)

    e_vals = np.array([quasi_energy(d, d) for d in tqdm(d_vals)])

    def eps_wf(a_m, c_m):
        theta_0 = phi(np.pi, d_vals, d_vals) / 2
        theta_m = np.sqrt(np.abs(a_m) ** 2 + np.abs(c_m) ** 2)
        sin_eps_pi = np.sin(theta_0) * np.cos(theta_m) + c_m * np.cos(theta_0) * sinx_over_x(theta_m)
        eps = np.arcsin(sin_eps_pi.astype(complex)).real / np.pi
        return eps

    # approximation for small field
    a1_m = np.array([a1_integral(v, 0, np.pi, d, d) for d in d_vals])
    c2_m = np.array([c2_integral(v, 0, np.pi, d, d) for d in d_vals])
    a3_m = np.array([a3_integral(v, 0, np.pi, d, d) for d in d_vals])

    plt.figure(figsize=(8, 6))


    # 0-the approximation
    phi_vals = phi(np.pi, d_vals, d_vals) / (2 * np.pi)  # at T/2
    eps_adiabatic_vals = (phi_vals + 1 / 2) % 1 - 1 / 2

    approx_0th = eps_wf(0, 0)
    approx_1st = eps_wf(a1_m, 0)
    approx_2nd = eps_wf(a1_m, c2_m)
    approx_3rd = eps_wf(a1_m + a3_m, c2_m)

    plt.plot(d_vals, approx_0th, '--', label='Adiabatic')
    # plt.plot(d_vals, approx_1st, '-.', label='1st Magnus')
    # plt.plot(d_vals, approx_2nd, ':', label='2nd Magnus')

    plt.plot(d_vals, approx_1st, ':', label='FMA')
    plt.plot(d_vals, approx_2nd, '-.', label='SMA')

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
    plt.savefig('figures/quasienergy/slow_driving/eps_magnus_symmetric_MA.pdf', dpi=400)
    plt.show()


if __name__ == '__main__':
    # adiabatic_limit()
    magnus_symmetric()

