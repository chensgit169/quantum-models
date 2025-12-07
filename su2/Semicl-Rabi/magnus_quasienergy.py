import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import j0
from tqdm import tqdm

from su2.common.test.root import find_roots
from su2.common.magnus.magnus_su2 import a3_integral, a1_integral, c2_integral
from exact_solution import quasi_energy

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


# limiting behavior for small gap
def small_gap(recompute=False, with_correction=False):
    data_filename = 'data/limiting/small_gap_demo_error.npz'
    if os.path.exists(data_filename) and not recompute:
        data = np.load(data_filename)
        f_vals = data['f']
        e_vals = data['e']
        v = data['v']
    else:
        v = 1.0
        f_vals = np.linspace(0, 20, 201)
        e_vals = np.array([quasi_energy(f, v, real_only=False).real for f in tqdm(f_vals)])
        np.savez(data_filename, f=f_vals, e=e_vals, v=v)

    line_main = plt.plot(f_vals, - e_vals, label='Exact')

    # plt.plot(f_vals,  e_vals, color =line_main[0].get_color())
    # plt.plot(f_vals, e_vals + 1, color=line_main[0].get_color())

    def f(t, _a, _d):
        return _d * np.exp(1j * _a * np.cos(t)) / 2

    approx_1st = v * j0(f_vals) / 2
    plt.plot(f_vals, approx_1st, '--k',
             label=r'$\Delta \ J_0(A)/2$')
    if with_correction:
        # compute third order correction
        integrals = [a3_integral(f, 0, 2 * np.pi, a, v) for a in f_vals]
        correction = np.array(integrals).real / (2 * np.pi)
        plt.plot(f_vals, approx_1st + correction, ':k',
                 label=r'$\Delta \ J_0(A)/2$ + 3rd order correction')

    # plt.figure(figsize=(6, 4))
    plt.xlabel('A')
    plt.ylabel(r'$\epsilon$')
    plt.xlim(np.min(f_vals), np.max(f_vals))
    # plt.title(r'$\Delta$=' + f'{v}')
    plt.axhline(0, color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    # plt.savefig('figures/limiting/small_gap_demo_error.pdf', dpi=400)
    plt.show()


# limiting behavior for weak field
def integral_result(omega, tol=1e-12):
    """
    Compute ∫_0^{2π} e^{i ω t} dt for omega (scalar or array).
    Supports arrays and handles the ω→0 limit safely.
    """
    res = np.zeros_like(omega, dtype=np.complex128)

    # for ω close to 0, use the limit value
    mask = np.abs(omega) < tol
    res[mask] = 2 * np.pi
    iw = 1j * omega[~mask]
    res[~mask] = (np.exp(iw * 2 * np.pi) - 1) / iw
    return res


def sinx_over_x(x):
    """Compute sin(x)/x safely for x near 0."""
    res = np.ones_like(x)
    mask = np.abs(x) > 1e-8
    res[mask] = np.sin(x[mask]) / x[mask]
    return res


# def int_f_d(d_vals):
#     return 1 / 1j * (integral_result(d_vals + 1) - integral_result(d_vals - 1))


def int_f_d(d_vals):
    return np.pi * (sinx_over_x(np.pi * (d_vals + 1)) - sinx_over_x(np.pi * (d_vals - 1)))


def demo_int_f_d():
    d_vals = np.linspace(0, 8, 2001)
    f_vals = int_f_d(d_vals) / (2 * np.pi)

    zeros = np.arange(8)

    plt.plot(d_vals, f_vals.real, label='Re')
    plt.plot(d_vals, f_vals.imag, label='Im')
    plt.plot(d_vals, np.zeros_like(d_vals), '--', color='gray')
    plt.plot(zeros, np.zeros_like(zeros), 'x', label='red')
    plt.xlabel(r'$\Delta$')
    plt.ylabel(r'$f(\Delta)$')
    plt.title(r'$f(\Delta)=\int_0^{2\pi} e^{i \Delta t} \sin(t) dt/2\pi$')
    plt.legend()
    plt.show()


def eps_wf(d_vals, a_m, c_m):
    theta_0 = np.pi * d_vals
    theta_m = np.sqrt(np.abs(a_m) ** 2 + np.abs(c_m) ** 2)
    approx = np.arccos(np.cos(theta_0) * np.cos(theta_m) - np.sin(theta_0) * np.cos(theta_m) * c_m) / (2 * np.pi)
    return approx


def weak_field(recompute=False, with_shift=False):
    a = 1.2

    if with_shift:
        data_filename = f'data/limiting/wf_avoided_crossing_a={a:.1f}.npz'
        img_filename = f'figures/limiting/wf_avoided_crossing_a={a:.1f}.pdf'
        d_vals = np.linspace(2.85, 3.05, 201)
    else:
        data_filename = f'data/limiting/weak_field_a={a:.1f}.npz'
        img_filename = f'figures/limiting/weak_field_a={a:.1f}.pdf'
        d_vals = np.linspace(0, 7.8, 201)

    if os.path.exists(data_filename) and not recompute:
        data = np.load(data_filename)
        d_vals = data['d']
        e_vals = data['e']
    else:
        # exact quasi-energy
        e_vals = np.array([quasi_energy(a, d) for d in tqdm(d_vals)])
        np.savez(data_filename, d=d_vals, e=e_vals, a=a)

    def f(t, _a, _d):
        return _a * np.sin(t) * np.exp(1j * _d * t) / 2

    # approximation for small field
    a1_m = np.array([a1_integral(f, 0, 2 * np.pi, a, d) for d in d_vals])
    c2_m = np.array([c2_integral(f, 0, 2 * np.pi, a, d) for d in d_vals])
    a3_m = np.array([a3_integral(f, 0, 2 * np.pi, a, d) for d in d_vals])

    approx_1st = eps_wf(d_vals, a1_m, 0)
    approx_2nd = eps_wf(d_vals, a1_m, c2_m)
    approx_3rd = eps_wf(d_vals, a1_m + a3_m, c2_m)

    line1 = plt.plot(d_vals, e_vals, label='Exact')
    color1 = line1[0].get_color()
    if not with_shift:
        plt.plot(d_vals, -e_vals, color=color1)

    plt.plot(d_vals, approx_1st, label='1st Magnus', linestyle='--', color='black')
    plt.plot(d_vals, approx_2nd, label='2nd Magnus', linestyle='-.', color='black')
    plt.plot(d_vals, approx_3rd, label='3rd Magnus', linestyle=':', color='black')

    if with_shift:
        peak = np.argmax(e_vals)
        d_peak = d_vals[peak]
        e_pos = e_vals[peak]
        print("d_peak =", d_peak, ", d_val =", e_pos)

        # Bloch-Siegert shift around d = 3
        delta_bs = a ** 2 / 8 / d_peak

        x0, y0 = d_peak, e_pos
        x1, y1 = d_peak, e_pos - delta_bs

        dx = x1 - x0
        dy = y1 - y0
        head_length = 0.005

        plt.arrow(x0, y0, dx, dy + head_length,
                  head_length=head_length, fc='red', ec='red', linewidth=0.25)

        plt.text(x0 - 0.01, (y0 + y1) / 2,
                 "$\\frac{A^2}{8\\Delta}$", fontsize=20, color="black", ha='center')

    plt.xlim(np.min(d_vals), np.max(d_vals))

    plt.grid(True, alpha=0.3)
    plt.xlabel(r'$\Delta$', fontsize=18)
    plt.ylabel(r'$\epsilon$', fontsize=20)
    plt.title(r'$A=$' + f'{a}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(img_filename, dpi=400)
    plt.show()


def fast_driving():
    a_vals = np.linspace(0, 1.0, 400)
    d_fixed = 0.05

    a1 = d_fixed / 2
    c2 = - (a_vals * d_fixed) / 2
    a3 = - 3 * (a_vals * d_fixed ** 2) / 8
    e_approx = np.sqrt((a1 + a3) ** 2 + c2 ** 2)

    e_vals = np.array([quasi_energy(a, d_fixed) for a in tqdm(a_vals)])

    line = plt.plot(a_vals, e_vals, label='Exact')
    color = line[0].get_color()
    plt.plot(a_vals, - e_vals, color=color)

    plt.plot(a_vals, e_approx, label='Approximation', linestyle='--')
    plt.xlabel('A')
    plt.ylabel(r'$\epsilon$')
    plt.axhline(0, color='gray', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # compute_zeros(recompute=False, illustration=True)
    # small_gap(with_correction=True)
    weak_field(recompute=False, with_shift=True)
    # demo_int_f_d()
    # fast_driving()
