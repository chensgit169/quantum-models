import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from exact_solution import quasi_energy
from su2.common.test.root import find_roots

plt.rcParams['font.size'] = 14
plt.rcParams['axes.titlesize'] = 25
plt.rcParams['axes.labelsize'] = 20


def compute_zeros(recompute=False, illustration=False):
    vs = np.linspace(-1e-4, 9, 400)

    zeros = []

    data_filename = 'data/zeros.npy'
    if os.path.exists(data_filename) and not recompute:
        zeros = np.load(data_filename)
    else:
        for v in tqdm(vs):
            def f(x):
                return quasi_energy(x, v)

            xs = np.linspace(-1e-4, 29, 1000)
            roots, _ = find_roots(xs, f)

            for r in roots:
                zeros.append((r, v))

        zeros = np.array(zeros)
        np.save(data_filename, zeros)

    line_main = plt.plot(zeros[:, 0], zeros[:, 1], 'x', markersize=2)
    color_main = line_main[0].get_color()
    fs = np.linspace(0, 29, 300)
    plt.plot(fs, np.zeros_like(fs), 'x', color=color_main)

    # bessel_zeros = jn_zeros(0, 9)
    # bessel_zeros = np.array(bessel_zeros)
    # plt.plot(bessel_zeros, np.zeros_like(bessel_zeros), 'o', label='Bessel zeros')
    # integers = np.arange(2, 10, 2)
    # plt.plot(np.zeros_like(integers), integers, '*', label='Integers')

    # plt.title('Quasi-energy zeros for semiclassical Rabi model')
    x_min, x_max = 0, 9
    y_min, y_max = 0, 9
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    if illustration:
        from matplotlib.patches import Rectangle, Patch
        import matplotlib.patheffects as pe

        ax = plt.gca()

        # # Background = remaining region (first quadrant)
        # bg = Rectangle((0, 0), x_max, y_max,
        #                facecolor='none', edgecolor='none', zorder=0)
        # ax.add_patch(bg)
        #
        # Region A: 0 < x < 1 (vertical strip) — forward-slash hatch
        rect_A = Rectangle((0, 0), 1, y_max,
                           facecolor='none', edgecolor='black',
                           hatch='/', linewidth=0.8, zorder=2)
        ax.add_patch(rect_A)

        # Region B: 0 < y < 1 (horizontal strip) — cross (x) hatch
        rect_B = Rectangle((0, 0), x_max, 1,
                           facecolor='none', edgecolor='black',
                           hatch='\\', linewidth=0.8, zorder=3)
        ax.add_patch(rect_B)

        # Boundary lines for x=1 and y=1
        ax.axvline(1, color='k', linestyle='--', zorder=5)
        ax.axhline(1, color='k', linestyle='--', zorder=5)

        # Legend
        legend_patches = [
            Patch(facecolor='none', edgecolor='black', hatch='//', label=r'$C_1$: 0 < $A$ < 1'),
            Patch(facecolor='none', edgecolor='black', hatch='\\\\', label=r'$C_2$: 0 < $\Delta$ < 1'),
            Patch(facecolor='none', edgecolor='black', label=r'$C_3$: $A$ > 1, $\Delta$ > 1'),
            # Patch(facecolor='none', edgecolor='black', hatch='xxx', label=r'Region $C_0$=$C_1$$\cap$$C_2$'),
        ]

        # === Text labels with white outline ===
        outline = [pe.Stroke(linewidth=6, foreground='white'), pe.Normal()]
        # ax.text(0.5, 0.5, r'$C_0$', ha='center', va='center', fontsize=16, fontweight='bold', path_effects=outline)
        ax.text(0.5, y_max / 2, r'$C_1$', ha='center', va='center', fontsize=16, fontweight='bold',
                path_effects=outline)
        ax.text(x_max / 2, 0.5, r'$C_2$', ha='center', va='center', fontsize=16, fontweight='bold',
                path_effects=outline)
        ax.text(x_max / 2, y_max / 2, r'$C_3$', ha='center', va='center', fontsize=16, fontweight='bold',
                path_effects=outline)

        ax.legend(handles=legend_patches, loc='upper right', fontsize=14)

    plt.xlabel('A')
    plt.ylabel(r'$\Delta$')
    plt.tight_layout()
    plt.savefig('figures/rabi/rabi_eps_zeros_wi.pdf', dpi=400)
    plt.show()
