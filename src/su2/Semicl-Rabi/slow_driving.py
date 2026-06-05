import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib
import numpy as np
import yaml
from scipy.special import ellipeinc
from tqdm import tqdm

from exact_solution import quasi_energy
from su2.common.utils import sinx_over_x
from su2.magnus import magnus_su2

from pathlib import Path

# path setting
file_path = Path(__file__).parent
data_path = file_path / "data" / "params.yaml"
img_folder = file_path / "figures" / "quasienergy" / "slow_driving"
if not img_folder.exists():
    img_folder.mkdir(parents=True)

mpl.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 24,
    "legend.fontsize": 21,
    "xtick.labelsize": 21,
    "ytick.labelsize": 21,
    "lines.linewidth": 3,
    "lines.markersize": 6,
    "axes.linewidth": 2,
    "figure.dpi": 300,
})


def load_params(which: int = 1):
    param_sets = yaml.safe_load(open(data_path, 'r'))['slow_driving']
    params = param_sets['params-set'+str(which)]
    g_min, g_max = params['g']['min'], params['g']['max']
    g_vals = np.linspace(g_min, g_max, 400)

    return g_vals


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


def eps_magnus(g, d):
    g = np.asarray(g)

    a1_m, c2_m, a3_m = magnus_su2(v_a, 0, np.pi, g, d, N=4000)
    a_m = a1_m + a3_m
    c_m = c2_m

    theta_0 = phi(np.pi, g, d) / 2
    theta_m = np.sqrt(np.abs(a_m) ** 2 + np.abs(c_m) ** 2)
    sin_eps_pi = np.sin(theta_0) * np.cos(theta_m) + c_m * np.cos(theta_0) * sinx_over_x(theta_m)
    eps = np.arcsin(sin_eps_pi.astype(complex)).real / np.pi
    return eps


def magnus_symmetric():
    g_vals = load_params(which=3)
    d = 1.0

    e_vals = np.array([quasi_energy(g, d) for g in tqdm(g_vals)])

    def eps_a(a_m, c_m):
        theta_0 = phi(np.pi, g_vals, d) / 2
        theta_m = np.sqrt(np.abs(a_m) ** 2 + np.abs(c_m) ** 2)
        sin_eps_pi = np.sin(theta_0) * np.cos(theta_m) + c_m * np.cos(theta_0) * sinx_over_x(theta_m)
        eps = np.arcsin(sin_eps_pi.astype(complex)).real / np.pi
        return eps

    a1_m, c2_m, a3_m = np.array([magnus_su2(v_a, 0, np.pi, g, d, N=4000) for g in g_vals]).T

    plt.figure(figsize=(8, 6))

    # 0-the approximation
    approx_0th = eps_a(0, 0)
    approx_1st = eps_a(a1_m, 0)
    approx_2nd = eps_a(a1_m, c2_m)
    approx_3rd = eps_a(a1_m + a3_m, c2_m)

    plt.plot(g_vals, approx_0th, '--', label='Adiabatic')
    # plt.plot(d_vals, approx_1st, '-.', label='1st Magnus')
    # plt.plot(d_vals, approx_2nd, ':', label='2nd Magnus')

    plt.plot(g_vals, approx_1st, ':', label='1st-order')
    plt.plot(g_vals, approx_2nd, '-.', label='2nd-order')
    plt.plot(g_vals, approx_3rd, '--', label='3rd-order')

    line1 = plt.plot(g_vals, e_vals, label='Exact', color='k')
    color1 = line1[0].get_color()
    plt.plot(g_vals, -e_vals, color=color1)

    # plt.plot(d_vals, approx_3rd, ':', label='3rd Magnus')
    # plt.grid(True, alpha=0.3)
    plt.title(r'$\Delta/\omega=$' + f'{d}')
    plt.xlabel(r'$g/\omega$')
    plt.ylabel(r'$\epsilon/\omega$')
    plt.legend(loc='best')
    plt.xlim(np.min(g_vals), np.max(g_vals))
    plt.tight_layout()
    plt.savefig(img_folder/'eps_magnus_symmetric_MA.pdf', dpi=400)
    plt.show()


def two_dim_plot():
    g_vals = np.linspace(0.01, 10, 100)
    d_vals = np.linspace(0.01, 10, 100)

    eps_m = np.array([[eps_magnus(g, d) for g in g_vals] for d in tqdm(d_vals)])
    eps_exact = np.array([[quasi_energy(g, d) for g in g_vals] for d in tqdm(d_vals)])

    xs, ys = np.meshgrid(g_vals, d_vals)

    # 统一前两个图的颜色范围
    vmin = min(eps_exact.min(), eps_m.min())
    vmax = max(eps_exact.max(), eps_m.max())

    # 误差
    diff = np.abs(eps_m/(1e-15+eps_exact) - 1)

    fig, axes = plt.subplots(3, 1, figsize=(7, 13))

    # --- Exact ---
    im0 = axes[0].pcolormesh(xs, ys, eps_exact,
                             shading='auto', cmap='viridis',
                             vmin=vmin, vmax=vmax)
    # axes[0].set_title("Exact")
    # axes[0].set_xlabel("g")
    # axes[0].set_ylabel("d")
    fig.colorbar(im0, ax=axes[0])

    # --- Magnus ---
    im1 = axes[1].pcolormesh(xs, ys, eps_m,
                             shading='auto', cmap='viridis',
                             vmin=vmin, vmax=vmax)
    # axes[1].set_title("Magnus Approx")
    # axes[1].set_xlabel("g")
    axes[1].set_ylabel(r"$\Delta/\omega$")
    fig.colorbar(im1, ax=axes[1])

    # --- Difference ---
    im2 = axes[2].pcolormesh(xs, ys, diff,
                             shading='auto', cmap='magma',
                             norm=matplotlib.colors.LogNorm())
    # axes[2].set_title("Absolute Error |Exact - Magnus|")
    axes[2].set_xlabel(r"$g/\omega$")

    axes[0].text(-0.4, 3.1, '(a)', fontsize=24, transform=axes[2].transAxes)
    axes[1].text(-0.4, 2.0, '(b)', fontsize=24, transform=axes[2].transAxes)
    axes[2].text(-0.4, 0.9, '(c)', fontsize=24, transform=axes[2].transAxes)
    # axes[2].set_ylabel("d")
    fig.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(img_folder/'magnus_2d_comparison.pdf', dpi=400)
    plt.show()


def magnus_symmetric_comparison():
    # Case 1: d = g
    g_vals = np.linspace(0.01, 4, 100)
    d_vals = g_vals.copy()  # d = g

    # Case 2: Fixed d = 1.0
    g_fixed = np.linspace(0.01, 10, 100)
    d_fixed = 1.0 * np.ones_like(g_fixed)

    def eps_a(a_m, c_m, g_vals_local, d_local):
        theta_0 = phi(np.pi, g_vals_local, d_local) / 2
        theta_m = np.sqrt(np.abs(a_m) ** 2 + np.abs(c_m) ** 2)
        sin_eps_pi = np.sin(theta_0) * np.cos(theta_m) + c_m * np.cos(theta_0) * sinx_over_x(theta_m)
        eps = np.arcsin(sin_eps_pi.astype(complex)).real / np.pi
        return eps

    # Create figure with two subplots (2 rows, 1 column)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 12))


    for ax, gs, ds in zip([ax1, ax2], [g_vals, g_fixed], [d_vals, d_fixed]):
        e_vals = np.array([quasi_energy(g, d, real_only=False).real for g, d in tqdm(zip(gs, ds), total=len(gs), desc="Exact")])

        a1, c2, a3 = np.array([
            magnus_su2(v_a, 0, np.pi, g, d, N=400) for g, d in tqdm(zip(gs, ds), total=len(gs))
        ]).T

        approx_0th = eps_a(0, 0, gs, ds)
        approx_1st = eps_a(a1, 0, gs, ds)
        approx_2nd = eps_a(a1, c2, gs, ds)
        approx_3rd = eps_a(a1 + a3, c2, gs, ds)

        ax.plot(gs, approx_0th, '--', label='Adiabatic')
        ax.plot(gs, approx_1st, ':', label='1st-order')
        ax.plot(gs, approx_2nd, '-.', label='2nd-order')
        ax.plot(gs[::3], approx_3rd[::3], 'o', label='3rd-order')

        line = ax.plot(gs, e_vals, label='Exact', color='k')
        ax.plot(gs, -e_vals, color=line[0].get_color())

        ax.set_ylabel(r'$\epsilon/\omega$', fontsize=24)

        ax.set_xlim(0, np.max(gs))

    ax1.text(0.05, 0.9, '(a)', fontsize=24, transform=ax1.transAxes)
    ax1.legend(loc='best')

    ax2.text(0.2, 0.9, '(b)', fontsize=24, transform=ax2.transAxes)

    ax2.set_xlabel(r'$g/\omega$', fontsize=24)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(img_folder/'magnus_symmetric_comparison.pdf', dpi=400)
    plt.show()


if __name__ == '__main__':
    # adiabatic_limit()
    # magnus_symmetric()
    # magnus_symmetric_comparison()
    two_dim_plot()
