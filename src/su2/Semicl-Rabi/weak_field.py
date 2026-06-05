import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import yaml
from tqdm import tqdm

from su2.common import sinx_over_x
from su2.magnus import magnus_su2
from exact_solution import quasi_energy

from pathlib import Path

# path setting
file_path = Path(__file__).parent
data_path = file_path / "data" / "params.yaml"
img_folder = file_path / "figures" / "quasienergy" / "weak_field"
if not img_folder.exists():
    img_folder.mkdir(parents=True)

mpl.rcParams.update({
    # "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 24,
    "legend.fontsize": 21,
    "xtick.labelsize": 24,
    "ytick.labelsize": 24,
    "lines.linewidth": 3,
    "lines.markersize": 6,
    "axes.linewidth": 2,
    "figure.dpi": 300,
})


def load_params(which: int = 1):
    param_sets = yaml.safe_load(open(data_path, 'r'))['weak_field']
    params = param_sets['params-set' + str(which)]
    g = params['g']
    d = params['d']
    d_min, d_max, num_d = d['min'], d['max'], d['N']
    d_vals = np.linspace(d_min, d_max, num_d)
    return g, d_vals


def v_g(t, _g, _d):
    return _g * np.sin(t) * np.exp(1j * _d * t) / 2


def demo_exact_eps():
    g = 1
    d_vals = np.linspace(0, 10, 201)
    e_vals = np.array([quasi_energy(g, d) for d in d_vals])

    plt.figure(figsize=(8, 6))
    line_main = plt.plot(d_vals, e_vals)
    plt.plot(d_vals, -1-e_vals, color=line_main[0].get_color())
    plt.ylim(-0.6, 0.5)
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
    plt.savefig(img_folder / 'exact_demo_g=1.pdf', dpi=300)
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
    g, d_vals = load_params(1)

    e_vals = np.array([quasi_energy(g, d) for d in tqdm(d_vals)])

    def eps_wf(a_m, c_m):
        c_m = c_m.real
        theta_0 = np.pi * d_vals
        theta_m = np.sqrt(np.abs(a_m) ** 2 + np.abs(c_m) ** 2)
        cos_pi_eps = np.cos(theta_0) * np.cos(theta_m) - np.sin(theta_0) * sinx_over_x(theta_m) * c_m
        approx = np.arccos(cos_pi_eps) / (2 * np.pi)
        return approx

    # approximation for small field
    a1_m, c2_m, a3_m = np.array([magnus_su2(v_g, -np.pi, np.pi, g, d) for d in d_vals]).T

    # print(np.max(np.abs(np.real(a1_m))))
    # print(np.max(np.abs(np.real(a3_m))))

    approx_1st = eps_wf(a1_m, 0)
    approx_2nd = eps_wf(a1_m, c2_m)
    approx_3rd = eps_wf(a1_m + a3_m, c2_m)
    plot_results(g, d_vals, e_vals, approx_1st, approx_2nd, approx_3rd)

    # from matplotlib.patches import Rectangle
    # rect = Rectangle((1.65, -0.1), 0.4, 0.2,
    #                  linewidth=1,
    #                  edgecolor='red',
    #                  facecolor='none',
    #                  alpha=0.8)
    # ax = plt.gca()
    # ax.add_patch(rect)

    plt.text(0.65, 0.4, '(a)', fontsize=24)

    plt.savefig(img_folder/f"direct_magnus_g={g}_avoided_crossing.pdf", dpi=400)
    plt.show()


def magnus_explicit_symmetry():
    def eps_wf(a_m, c_m):
        theta_0 = np.pi * d_vals / 2
        theta_m = np.sqrt(np.abs(a_m) ** 2 + np.abs(c_m) ** 2)
        sin_eps_pi = np.sin(theta_0) * np.cos(theta_m) + c_m * np.cos(theta_0) * sinx_over_x(theta_m)
        eps = np.arcsin(sin_eps_pi.astype(complex)).real / np.pi
        return eps

    g, d_vals = load_params(1)
    e_vals = np.array([quasi_energy(g, d) for d in tqdm(d_vals)])

    # approximation for small field
    a1_m, c2_m, a3_m = np.array([magnus_su2(v_g, 0, np.pi, g, d) for d in d_vals]).T
    c2_m = c2_m.real

    approx_1st = eps_wf(a1_m, 0)
    approx_2nd = eps_wf(a1_m, c2_m)

    approx_3rd = eps_wf(a1_m + a3_m, c2_m)
    print(abs(approx_2nd - approx_3rd).max())

    plot_results(g, d_vals, e_vals, approx_1st, approx_2nd, approx_3rd)

    plt.savefig(img_folder/f"explicit_symmetric_g={g}_MA.pdf", dpi=400)
    plt.show()


def compare_magnus():
    def eps_wf_direct(a_m, c_m, d_vals):
        theta_0 = np.pi * d_vals
        theta_m = np.sqrt(np.abs(a_m) ** 2 + np.abs(c_m) ** 2)
        cos_pi_eps = np.cos(theta_0) * np.cos(theta_m) - np.sin(theta_0) * sinx_over_x(theta_m) * c_m
        approx = np.arccos(cos_pi_eps) / (2 * np.pi)
        return approx

    def eps_wf_symmetric(a_m, c_m, d_vals):
        theta_0 = np.pi * d_vals / 2
        theta_m = np.sqrt(np.abs(a_m) ** 2 + np.abs(c_m) ** 2)
        sin_eps_pi = np.sin(theta_0) * np.cos(theta_m) + c_m * np.cos(theta_0) * sinx_over_x(theta_m)
        eps = np.arcsin(sin_eps_pi.astype(complex)).real / np.pi
        return eps

    g, d_vals = load_params(1)
    e_vals = np.array([quasi_energy(g, d) for d in tqdm(d_vals)])

    # Create figure with two subplots (2 rows, 1 column)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 12))

    # ========== Upper subplot: direct_magnus ==========
    a1_m, c2_m, a3_m = np.array([magnus_su2(v_g, -np.pi, np.pi, g, d) for d in d_vals]).T
    c2_m = c2_m.real

    approx_1st = eps_wf_direct(a1_m, 0, d_vals)
    approx_2nd = eps_wf_direct(a1_m, c2_m, d_vals)
    approx_3rd = eps_wf_direct(a1_m + a3_m, c2_m, d_vals)

    ax1.plot(d_vals, approx_1st, '--', label='1st-order')
    ax1.plot(d_vals, approx_2nd, ':', label='2nd-order')
    ax1.plot(d_vals, approx_3rd, '-.', label='3rd-order')
    line_exact = ax1.plot(d_vals, e_vals, 'k', label='Exact')
    ax1.plot(d_vals, -e_vals, color=line_exact[0].get_color())
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax1.set_ylabel(r'$\epsilon/\omega$', fontsize=24)
    ax1.legend(loc='lower left')
    ax1.text(0.05, 0.9, '(a)', fontsize=24, transform=ax1.transAxes)
    ax1.set_xlim(np.min(d_vals), np.max(d_vals))

    # ========== Lower subplot: magnus_explicit_symmetry ==========
    a1_m_sym, c2_m_sym, a3_m_sym = np.array([magnus_su2(v_g, 0, np.pi, g, d) for d in d_vals]).T

    approx_1st_sym = eps_wf_symmetric(a1_m_sym, 0, d_vals)
    approx_2nd_sym = eps_wf_symmetric(a1_m_sym, c2_m_sym, d_vals)
    approx_3rd_sym = eps_wf_symmetric(a1_m_sym + a3_m_sym, c2_m_sym, d_vals)

    ax2.plot(d_vals, approx_1st_sym, '--', label='1st-order')
    ax2.plot(d_vals, approx_2nd_sym, ':', label='2nd-order')
    ax2.plot(d_vals, approx_3rd_sym, '-.', label='3rd-order')
    line_exact = ax2.plot(d_vals, e_vals, 'k', label='Exact', linewidth=2)
    ax2.plot(d_vals, -e_vals, color=line_exact[0].get_color(), linewidth=2)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.3)
    ax2.set_xlabel(r'$\Delta/\omega$', fontsize=24)
    ax2.set_ylabel(r'$\epsilon/\omega$', fontsize=24)
    # ax2.legend(loc='best')
    ax2.text(0.05, 0.9, '(b)', fontsize=24, transform=ax2.transAxes)
    ax2.set_xlim(np.min(d_vals), np.max(d_vals))

    plt.tight_layout()
    plt.savefig(img_folder / "compare_magnus_g=1.pdf", dpi=400)
    plt.show()


def demo_avoided_crossing():
    g = 1

    # Create figure with two subplots (2 rows, 1 column)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))

    # ========== Upper subplot (exact_eps) ==========
    d_vals_full = np.linspace(0, 10, 201)
    e_vals_full = np.array([quasi_energy(g, d) for d in d_vals_full])

    line_main = ax1.plot(d_vals_full, e_vals_full)
    ax1.plot(d_vals_full, -1 - e_vals_full, color=line_main[0].get_color())
    ax1.plot(d_vals_full, - e_vals_full, color=line_main[0].get_color())
    ax1.plot(d_vals_full, -1 + e_vals_full, color=line_main[0].get_color())
    ax1.set_ylim(-0.6, 0.5)
    ax1.set_ylabel(r'$\epsilon/\omega$')
    ax1.set_xlim(np.min(d_vals_full), np.max(d_vals_full))
    ax1.hlines(y=-0.5, xmin=np.min(d_vals_full), xmax=np.max(d_vals_full),
               colors='gray', linestyles='dashed', alpha=0.5)

    from matplotlib.patches import Rectangle
    rect = Rectangle((2.7, -0.45), 0.4, -0.1,
                     linewidth=2, edgecolor='red',
                     facecolor='none', alpha=0.8)
    ax1.add_patch(rect)
    ax1.text(1.8, 0.4, '(a)', fontsize=24)

    # ========== Lower subplot (avoided_crossing) ==========
    d_vals_zoom = np.linspace(2.85, 2.96, 1001)
    e_vals_zoom = np.array([quasi_energy(g, d) for d in d_vals_zoom])

    line_zoom = ax2.plot(d_vals_zoom, e_vals_zoom)
    ax2.plot(d_vals_zoom, -1 - e_vals_zoom, color=line_zoom[0].get_color())
    ax2.hlines(y=-0.5, xmin=np.min(d_vals_zoom), xmax=np.max(d_vals_zoom),
               colors='gray', linestyles='dashed', alpha=0.5)
    ax2.set_xlabel(r'$\Delta/\omega$')
    ax2.set_ylabel(r'$\epsilon/\omega$')
    ax2.set_xlim(np.min(d_vals_zoom), np.max(d_vals_zoom))
    ax2.text(2.875, -0.48, '(b)', fontsize=24)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(img_folder/'exact_demo_avoided_crossing_g=1.pdf', dpi=300)
    plt.show()


if __name__ == '__main__':
    # weak_field(recompute=True)
    # demo_exact_eps()
    # demo_avoided_crossing()

    # direct_magnus()
    compare_magnus()
    # magnus_explicit_symmetry()
