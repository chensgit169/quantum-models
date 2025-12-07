from su2.common.adiabatic.integrator import evolve
from su2.common.bloch import make_movie, plot_xy_trajectory

import matplotlib.pyplot as plt
import numpy as np

import os

plt.rcParams['font.size'] = 14

# d = 1
v = 0.01
g = 1 / (2 * v)
p_exact = np.exp(-2 * np.pi * g)


def E(t): return -v


def A(t): return v * t


data_filename = f'data/d=1_v={v:.3f}.npz'

recompute = True

if not os.path.exists(data_filename) or recompute:
    ts = np.linspace(-20, 30, 100000) / np.sqrt(v)

    data = evolve(ts, A, E)

    alpha_vals, beta_vals = data['psi_t'].T
    phi_vals = data['phi']

    np.savez_compressed(data_filename,
                        ts=ts,
                        alpha_vals=alpha_vals,
                        beta_vals=beta_vals,
                        phi_vals=phi_vals)
else:
    data = np.load(data_filename)
    ts = data['ts']
    alpha_vals = data['alpha_vals']
    beta_vals = data['beta_vals']
    phi_vals = data['phi_vals']

ts_demo = ts * np.sqrt(v)
demo_mask = (ts_demo >= 24) & (ts_demo <= 200)


phi_wrapped = np.angle(np.exp(-2j * phi_vals)/1j)


# # plt.plot(ts, beta_vals.real, label='Re')
# # plt.plot(ts, beta_vals.imag, label='Im')
#
#
# plt.plot(ts, np.abs(beta_vals) ** 2, label=r'$|\beta(t)|^2$')
# plt.axhline(p_exact, color='k', linestyle='--', label='Exact $P_{LZ}$')

def omega(t):
    return np.sqrt(1 ** 2 + A(t) ** 2)


def demo_phase():
    phase = np.angle(beta_vals / alpha_vals)
    plt.figure(figsize=(8, 5))
    plt.plot(ts_demo[demo_mask], phase[demo_mask], label=f'numerics')
    plt.plot(ts_demo[demo_mask], phi_wrapped[demo_mask], '--', label='φ(t)')
    diff = phase - phi_wrapped
    # plt.plot(ts_demo[demo_mask], diff[demo_mask], ':', label='Phase Difference')
    plt.xlabel(r'$\tau=\sqrt{v}r$')
    plt.ylabel(r'$\phi$(mod $2\pi$)')
    plt.title(r'Time Evolution of Phase $\beta(t)/\alpha(t)$')
    plt.legend()
    plt.grid(True)
    plt.show()


def demo_prob():
    plt.figure(figsize=(8, 5))
    plt.plot(ts_demo, np.abs(beta_vals) ** 2, label=r'$|\beta(t)|^2$')
    plt.axhline(p_exact, color='k', linestyle='--', label='Exact $P_{LZ}$')
    plt.xlabel(r'$\tau=\sqrt{v}r$')
    plt.ylabel(r'$\beta(t)$')
    plt.title(f'Time Evolution of β(t) at p=0, g={g}')
    plt.legend()
    plt.grid(True)
    plt.show()


def demo_coeff():
    plt.figure(figsize=(8, 5))
    plt.plot(ts_demo[demo_mask], beta_vals.real[demo_mask], label='Re[β(t)]')
    plt.plot(ts_demo[demo_mask], beta_vals.imag[demo_mask], label='Im[β(t)]')
    plt.xlabel(r'$\tau=\sqrt{v}r$')
    plt.ylabel(r'$\beta(t)$')
    plt.title(f'Time Evolution of β(t) at p=0, g={g}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def make_bloch_movie():
    psi = np.array([alpha_vals, beta_vals]).T
    movie_filename = f'figures/LZ_d=1_v={v:.3f}.gif'
    make_movie(psi, movie_filename, N=range(0, len(ts), 2000), theta_lim=np.pi / 3)


def show_xy_trajectory():
    psi = np.array([alpha_vals, beta_vals]).T
    plot_xy_trajectory(psi)

    scale = 2e-5
    plt.xlim([-scale, scale])
    plt.ylim([-scale, scale])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # demo_prob()
    demo_phase()
    # demo_coeff()
    # make_bloch_movie()
    # show_xy_trajectory()
