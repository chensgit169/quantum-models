import matplotlib.pyplot as plt
import numpy as np
import yaml
from tqdm import tqdm

from exact_solution import quasi_energy
from su2.common.magnus.magnus_su2 import a3_integral, a1_integral, c2_integral, sinx_over_x

plt.rcParams['font.size'] = 14
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.labelsize'] = 21


param_sets = yaml.safe_load(open('data/params.yaml', 'r'))['weak_field']


def v(t, _a, _d):
    return _a * np.sin(t) * np.exp(1j * _d * t) / 2


def demo_exact_eps():
    g = 1
    d_vals = np.linspace(0, 10, 201)
    e_vals = np.array([quasi_energy(g, d) for d in d_vals])

    plt.figure(figsize=(8, 6))
    plt.plot(d_vals, e_vals)
    plt.xlabel(r'$\Delta/\omega$')
    plt.ylabel(r'$\epsilon/\omega$')
    # plt.title(r'Quasienergy for $g=$'+f'{g}')
    plt.xlim(np.min(d_vals), np.max(d_vals))
    plt.hlines(y=-0.5, xmin=np.min(d_vals), xmax=np.max(d_vals), colors='gray', linestyles='dashed', alpha=0.5)

    from matplotlib.patches import Rectangle
    rect = Rectangle((2.7, -0.45), 0.4, -0.1,
                     linewidth=1,
                     edgecolor='red',
                     facecolor='none',
                     alpha=0.8)

    ax = plt.gca()
    ax.add_patch(rect)

    plt.text(2, 0.4, '(a)', fontsize=24)
    # plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/quasienergy/exact_demo/demo_quasienergy_g=1.pdf', dpi=300)
    plt.show()


def demo_avoided_crossing():
    g = 1
    plt.figure(figsize=(8, 6))
    d_vals = np.linspace(2.85, 2.96, 1001)
    e_vals = np.array([quasi_energy(g, d) for d in d_vals])
    plt.plot(d_vals, e_vals)
    plt.hlines(y=-0.5, xmin=np.min(d_vals), xmax=np.max(d_vals), colors='gray', linestyles='dashed', alpha=0.5)
    plt.xlabel(r'$\Delta/\omega$')
    plt.ylabel(r'$\epsilon/\omega$')
    # plt.title(r'Quasienergy for $g=$'+f'{g} (avoided crossing)')
    plt.xlim(np.min(d_vals), np.max(d_vals))
    # plt.ylim(-0.5, 0.5)
    plt.text(2.885, -0.48, '(b)', fontsize=24)
    # plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig('figures/quasienergy/exact_demo/demo_avoided_crossing_g=1.pdf', dpi=300)
    plt.show()


def plot_results(g, d_vals, e_vals, approx_1st, approx_2nd, approx_3rd):
    plt.figure(figsize=(8, 6))

    plt.plot(d_vals, approx_1st, label='FMA', linestyle='--')
    plt.plot(d_vals, approx_2nd, label='SMA', linestyle=':')
    plt.plot(d_vals, approx_3rd, label='TMA', linestyle='-.')

    line1 = plt.plot(d_vals, e_vals, label='Exact', color='k')
    color1 = line1[0].get_color()
    plt.plot(d_vals, -e_vals, color=color1)

    plt.axhline(0, color='gray', linestyle='--', alpha=0.3)
    plt.xlim(np.min(d_vals), np.max(d_vals))

    # plt.grid(True, alpha=0.3)
    plt.xlabel(r'$\Delta/\omega$')
    plt.ylabel(r'$\epsilon/\omega$')
    # plt.title(r'$g=$' + f'{g}')
    plt.legend(loc='best')
    plt.tight_layout()


def direct_magnus():
    params = param_sets['params-set2']
    g = params['g']
    d = params['d']
    d_min, d_max, num_d = d['min'], d['max'], d['N']
    d_vals = np.linspace(d_min, d_max, num_d)

    e_vals = np.array([quasi_energy(g, d) for d in tqdm(d_vals)])

    def eps_wf(a_m, c_m):
        theta_0 = np.pi * d_vals
        theta_m = np.sqrt(np.abs(a_m) ** 2 + np.abs(c_m) ** 2)
        cos_pi_eps = np.cos(theta_0) * np.cos(theta_m) - np.sin(theta_0) * sinx_over_x(theta_m) * c_m
        approx = np.arccos(cos_pi_eps) / (2 * np.pi)
        return approx

    # approximation for small field
    a1_m = np.array([a1_integral(v, -np.pi, 1 * np.pi, g, d) for d in d_vals])
    c2_m = np.array([c2_integral(v, -np.pi, 1 * np.pi, g, d) for d in d_vals])
    a3_m = np.array([a3_integral(v, -np.pi, 1 * np.pi, g, d) for d in d_vals])

    print(np.max(np.abs(np.real(a1_m))))
    print(np.max(np.abs(np.real(a3_m))))

    approx_1st = eps_wf(a1_m, 0)
    approx_2nd = eps_wf(a1_m, c2_m)
    approx_3rd = eps_wf(a1_m + a3_m, c2_m)
    plot_results(g, d_vals, e_vals, approx_1st, approx_2nd, approx_3rd)

    from matplotlib.patches import Rectangle
    rect = Rectangle((1.65, -0.1), 0.4, 0.2,
                     linewidth=1,
                     edgecolor='red',
                     facecolor='none',
                     alpha=0.8)
    ax = plt.gca()
    # ax.add_patch(rect)

    # plt.text(0.65, 0.4, '(a)', fontsize=24)

    plt.savefig("figures/quasienergy/weak_field/"+
                f"direct_magnus_g={g}_avoided_crossing.pdf", dpi=400)
    plt.show()


def magnus_explicit_symmetry():
    def eps_wf(a_m, c_m):
        theta_0 = np.pi * d_vals / 2
        theta_m = np.sqrt(np.abs(a_m) ** 2 + np.abs(c_m) ** 2)
        sin_eps_pi = np.sin(theta_0) * np.cos(theta_m) + c_m * np.cos(theta_0) * sinx_over_x(theta_m)
        eps = np.arcsin(sin_eps_pi.astype(complex)).real / np.pi
        return eps

    params = param_sets['params-set1']
    g = params['g']
    d = params['d']
    d_min, d_max, num_d = d['min'], d['max'], d['N']

    d_vals = np.linspace(d_min, d_max, num_d)

    e_vals = np.array([quasi_energy(g, d) for d in tqdm(d_vals)])

    # approximation for small field
    a1_m = np.array([a1_integral(v, 0, np.pi, g, d) for d in d_vals])
    c2_m = np.array([c2_integral(v, 0, np.pi, g, d) for d in d_vals])
    a3_m = np.array([a3_integral(v, 0, np.pi, g, d) for d in d_vals])

    approx_1st = eps_wf(a1_m, 0)
    approx_2nd = eps_wf(a1_m, c2_m)

    approx_3rd = eps_wf(a1_m + a3_m, c2_m)
    print(abs(approx_2nd - approx_3rd).max())

    plot_results(g, d_vals, e_vals, approx_1st, approx_2nd, approx_3rd)

    # plt.text(0.65, 0.4, '(b)', fontsize=24)
    plt.savefig("figures/quasienergy/weak_field/"
                + f"explicit_symmetric_g={g}_MA.pdf", dpi=400)
    plt.show()


if __name__ == '__main__':
    # weak_field(recompute=True)
    # demo_exact_eps()
    # demo_avoided_crossing()
    # direct_magnus()
    magnus_explicit_symmetry()
