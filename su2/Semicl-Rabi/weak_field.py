import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import yaml

from exact_solution import quasi_energy
from su2.common.magnus.magnus_su2 import a3_integral, a1_integral, c2_integral

plt.rcParams['font.size'] = 14


param_sets = yaml.safe_load(open('data/params.yaml', 'r'))['weak_field']


def eps_wf(d_vals, a_m, c_m):
    theta_0 = np.pi * d_vals / 2
    theta_m = np.sqrt(np.abs(a_m) ** 2 + np.abs(c_m) ** 2)
    sin_eps_pi = np.sin(theta_0) * np.cos(theta_m) + c_m * np.cos(theta_0) * np.sin(theta_m) / theta_m
    eps = np.arcsin(sin_eps_pi.astype(complex)).real / np.pi
    return eps


def weak_field(recompute=False):
    params = param_sets['params-set2']
    a = params['a']
    d = params['d']
    d_min, d_max, num_d = d['min'], d['max'], d['N']

    d_vals = np.linspace(d_min, d_max, num_d)

    def v(t, _a, _d):
        return _a * np.sin(t) * np.exp(1j * _d * t) / 2

    e_vals = np.array([-quasi_energy(a, d) for d in tqdm(d_vals)])

    # approximation for small field
    a1_m = np.array([a1_integral(v, 0, np.pi, a, d) for d in d_vals])
    c2_m = np.array([c2_integral(v, 0, np.pi, a, d) for d in d_vals])
    a3_m = np.array([a3_integral(v, 0, np.pi, a, d) for d in d_vals])

    approx_1st = eps_wf(d_vals, a1_m, 0)
    approx_2nd = eps_wf(d_vals, a1_m, c2_m)
    approx_3rd = eps_wf(d_vals, a1_m + a3_m, c2_m)

    line1 = plt.plot(d_vals, e_vals, label='Exact')
    color1 = line1[0].get_color()

    # plt.plot(d_vals, approx_1st, label='1st Magnus', linestyle='--', color='black')
    plt.plot(d_vals, approx_2nd, label='2nd Magnus', linestyle='-.', color='black')
    plt.plot(d_vals, approx_3rd, label='3rd Magnus', linestyle=':', color='black')

    plt.xlim(np.min(d_vals), np.max(d_vals))

    plt.grid(True, alpha=0.3)
    plt.xlabel(r'$\Delta$', fontsize=18)
    plt.ylabel(r'$\epsilon$', fontsize=20)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    plt.title(r'weak field $A=$' + f'{a}')
    # plt.ylim(-0.5, 0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('weak_field_avoided_crossing.png', dpi=300)
    plt.show()


if __name__ == '__main__':
    weak_field(recompute=True)