import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.special import j0
from tqdm import tqdm

from exact_solution import quasi_energy
from su2.common.utils import sinx_over_x
from su2.magnus import magnus_su2

plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 21


def load_params(which: int = 1):
    param_sets = yaml.safe_load(open('data/params.yaml', 'r'))['small_gap']

    params = param_sets['params-set'+str(which)]
    g = params['g']
    d = params['d']
    g_min, g_max, num_g = g['min'], g['max'], g['N']

    g_vals = np.linspace(g_min, g_max, num_g)
    return d, g_vals


def v_d(t, g, d):
    return d * np.exp(1j * g * np.sin(t)) / 2


def demo_exact_eps():
    g_vals = np.linspace(-0.8, 0.8, 201)

    ds = [0.9, 1.0, 1.1]
    styles = ['--', '-', ':']
    for d, style in zip(ds, styles):
        e_vals = np.array([quasi_energy(g, d, real_only=False).real for g in g_vals])
        line_mian = plt.plot(g_vals, e_vals, style, label=r'$\Delta/\omega$' + f'={d}')
        plt.plot(g_vals, 1 - e_vals, style, color=line_mian[0].get_color())

    plt.xlabel(r'$g/\omega$')
    plt.ylabel(r'$\epsilon/\omega$')
    # plt.title(r'Exact Rabi Quasienergy for $\Delta=$' + f'{d}')
    # plt.xlim(np.min(d_vals), np.max(d_vals))
    # plt.hlines(y=-0.5, xmin=np.min(d_vals), xmax=np.max(d_vals), colors='gray', linestyles='dashed', alpha=0.5)

    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/quasienergy/small_gap/exact_demo.pdf")
    plt.show()


def direct_magnus():
    d, g_vals = load_params()
    e_vals = np.array([quasi_energy(f, d, real_only=False).real for f in tqdm(g_vals)])

    line_main = plt.plot(g_vals, e_vals, label='Exact')
    plt.plot(g_vals, -e_vals, color=line_main[0].get_color())

    # plt.plot(f_vals, e_vals + 1, color=line_main[0].get_color())

    def eps_sg(a_m, c_m):
        theta_m = np.sqrt(np.abs(a_m) ** 2 + np.abs(c_m) ** 2)
        approx = np.arccos(np.cos(theta_m)) / (2 * np.pi)
        return approx

    # compute third order correction
    a1_m, c2_m, a3_m = np.array([magnus_su2(v_d, 0, 2 * np.pi, g, d) for g in g_vals]).T

    # approx_1st = d * j0(g_vals) / 2

    approx_1st = eps_sg(a1_m, 0)
    approx_2nd = eps_sg(a1_m, c2_m)
    approx_3rd = eps_sg(a1_m + a3_m, c2_m)
    plt.axhline(0.5, color='gray', linestyle='--')

    plt.plot(g_vals, d / 2 * j0(g_vals),
             '--', label=r'$\frac{\Delta}{2}J_0(\frac{g}{\omega})$')
    plt.plot(g_vals, approx_1st, '--', label=r'FMA')
    plt.plot(g_vals, approx_2nd, ':', label=r'SMA')
    plt.plot(g_vals, approx_3rd, '-.', label=r'TMA')

    # plt.figure(figsize=(6, 4))
    plt.xlabel(r'$g/\omega$')
    plt.ylabel(r'$\epsilon/\omega$')
    plt.xlim(np.min(g_vals), np.max(g_vals))
    # plt.title(r'$\Delta$=' + f'{d}')
    plt.axhline(0, color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/quasienergy/small_gap/divergent_demo.pdf', dpi=400)
    plt.show()


def magnus_explicit_symmetry(show_minus_eps=False):
    d, g_vals = load_params()

    e_vals = np.array([quasi_energy(f, d, real_only=False).real for f in tqdm(g_vals)])

    def eps_sg(a_m, c_m):
        theta_m = np.sqrt(np.abs(a_m) ** 2 + np.abs(c_m) ** 2)
        r = np.real(a_m) * sinx_over_x(theta_m)
        return np.arcsin(r) / np.pi

    # approximation for small field
    a1_m, c2_m, a3_m = np.array([magnus_su2(v_d, 0, np.pi, g, d) for g in g_vals]).T

    # check the imaginary part of A_n
    # print(a1_m)
    # print(np.max(np.abs(np.imag(a1_m))))
    # print(np.max(np.abs(np.imag(a3_m))))

    approx_1st = eps_sg(a1_m, 0)
    approx_2nd = eps_sg(a1_m, c2_m)
    approx_3rd = eps_sg(a1_m + a3_m, c2_m)

    plt.plot(g_vals, d / 2 * j0(g_vals),
             '--', label=r'$\frac{\Delta}{2}J_0(\frac{g}{\omega})$')
    # plt.plot(g_vals, approx_1st, '--', label=r'FMA')
    plt.plot(g_vals, approx_2nd, ':', label=r'FMA=SMA')
    plt.plot(g_vals, approx_3rd, '-.', label=r'TMA')
    # plt.axhline(0, color='gray', linestyle='--')

    line_main = plt.plot(g_vals, e_vals, label='Exact', color='k')
    if show_minus_eps:
        plt.plot(g_vals, -e_vals, color=line_main[0].get_color())

    # plt.plot(g_vals, 1 - e_vals, color=line_main[0].get_color())
    # plt.grid(True, alpha=0.3)

    plt.xlim(np.min(g_vals), np.max(g_vals))
    plt.xlabel(r'$g/\omega$', fontsize=21)
    plt.ylabel(r'$\epsilon/\omega$', fontsize=21)
    # plt.title(r'Rabi Quasienergy $\Delta=$' + f'{d}')
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/quasienergy/small_gap/"
                + f"explicit_symmetric_d={d}_MA.pdf", dpi=400)
    plt.show()


if __name__ == '__main__':
    # direct_magnus()
    magnus_explicit_symmetry(show_minus_eps=True)
    # demo_exact_eps()
