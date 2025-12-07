import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from i_integrals import IR_t_r, II_t_r, IR_t_r_derivative, I_beta, I_beta_derivative
from correlation import correlations

A0 = 1.0
r_vals = np.linspace(0, 20.0, 400)
t_vals = np.linspace(0, 10.0, 400)

plt.rcParams['font.size'] = 16


def compute(name: str = 'IR', recompute: bool = False):
    func = {'IR': IR_t_r, 'II': II_t_r, 'IR_d': IR_t_r_derivative}[name]

    data_filename = f'data/{name}_t_r_numeric_A0={A0:.1f}.npz'
    if os.path.exists(data_filename) and not recompute:
        data = np.load(data_filename)
        R = data['R']
        T = data['T']
        int_vals = data[f'{name}_vals']
    else:
        R, T = np.meshgrid(r_vals, t_vals)
        int_vals = np.zeros((len(t_vals), len(r_vals)), dtype=complex)

        for i, t in enumerate(tqdm(t_vals)):
            for j, r in enumerate(r_vals):
                int_vals[i, j] = func(t, r, A0)

        np.savez(data_filename, R=R, T=T, **{f'{name}_vals': int_vals})

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(R, T, np.abs(int_vals))
    plt.colorbar(label=f'|${name}$|')
    plt.xlabel('r')
    plt.ylabel('t')
    plt.show()


def compute_beta(recompute: bool = False):
    data_filename = f'data/I_beta_numeric_A0={A0:.1f}.npz'
    if not os.path.exists(data_filename) or recompute:
        I_beta_vals = np.array([I_beta(t, A0) for t in tqdm(t_vals)])
        mip_I_beta_vals = np.array([I_beta_derivative(t, A0) for t in tqdm(t_vals)])
        np.savez(data_filename, t_vals=t_vals, I_beta_vals=I_beta_vals, mip_I_beta_vals=mip_I_beta_vals)
    else:
        data = np.load(data_filename)
        I_beta_vals = data['I_beta_vals']
        mip_I_beta_vals = data['mip_I_beta_vals']

    plt.figure(figsize=(8, 6))
    plt.plot(t_vals, np.abs(I_beta_vals), label=r'|$I_\beta$|')
    plt.plot(t_vals, np.abs(mip_I_beta_vals), label=r"|$dI_\beta/dt$|")
    plt.xlim(left=0)
    plt.xlabel('t')
    # plt.ylabel('|Value|')
    plt.legend()
    plt.tight_layout()
    plt.show()


def correlation_heatmap():
    A0 = 1.0
    data_IR = np.load(f'data/IR_t_r_numeric_A0={A0:.1f}.npz')
    data_II = np.load(f'data/II_t_r_numeric_A0={A0:.1f}.npz')
    data_IR_d = np.load(f'data/IR_d_t_r_numeric_A0={A0:.1f}.npz')

    data_beta = np.load(f'data/I_beta_numeric_A0={A0:.1f}.npz')

    R = data_IR['R']
    T = data_IR['T']

    r_lim = 1.0
    cols = r_vals >= r_lim

    IR_vals = data_IR[f'IR_vals']
    II_vals = data_II[f'II_vals']
    IR_d_vals = data_IR_d[f'IR_d_vals']
    Ibeta_vals = data_beta['I_beta_vals'][None, :]
    mip_Ibeta_vals = data_beta['mip_I_beta_vals'][None, :]

    # cc_vals = -np.abs(IR_vals)**2 + np.abs(II_vals)**2 + np.abs(IR_d_vals)**2
    pi00, pi11 = correlations(R, IR_vals, IR_d_vals, II_vals, Ibeta_vals, mip_Ibeta_vals)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    # sub-plot 1：pi00
    im0 = axes[0].pcolormesh(R[:, cols], T[:, cols], pi00[:, cols],
                             shading='auto', rasterized=True)
    fig.colorbar(im0, ax=axes[0])  # , label=r'|$\Pi_{00}|$')
    axes[0].set_xlabel("r")
    axes[0].set_ylabel("t")
    axes[0].set_xticks(range(1, 20, 3))
    axes[0].set_title(r"$\Pi_{00}(r,t)$")

    # sub-plot 2：pi11
    im1 = axes[1].pcolormesh(R[:, cols], T[:, cols], pi11[:, cols],
                             shading='auto', rasterized=True)
    fig.colorbar(im1, ax=axes[1])  # , label=r'|$\Pi_{11}|$')
    axes[1].set_xlabel("r")
    axes[1].set_xticks(range(1, 20, 3))
    axes[1].set_title(r"$\Pi_{11}(r,t)$")

    plt.tight_layout()
    plt.savefig(f'figures/correlation_A0={A0:.1f}.pdf')
    plt.show()


if __name__ == '__main__':
    # draw_heatmap('IR_d', recompute=True)
    correlation_heatmap()
    # compute_beta()
