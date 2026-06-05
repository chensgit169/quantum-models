from su2.common.adiabatic.integrator import evolve

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.constants import m_e, eV, c

import os

from sauter_pulse import sauter_pulse, demo_p_dependence, load_double_sauter
from double_sauter import p_spectrum
"""
Demonstrations of dynamically assisted Schwinger effect using double Sauter pulses.

The following parameters and figure labels are aligned with article 
M. Orthaber, F. Hebenstreit, R. Alkofer, Phys. Lett. B 698 (2011) 80-85.
https://doi.org/10.1016/j.physletb.2011.02.053.


Chen
Last modified: 2025-9-16
"""

plt.rcParams['font.size'] = 14

# electron mass in eV, 510998.9499961642
m = m_e * c ** 2 / eV

E1 = 0.25
tau1_eV = 1e-4  # in unit of eV
tau1 = tau1_eV * m  # in unit of 1/m
E2 = 0.025
tau2_eV = 2e-6  # in unit of eV
tau2 = tau2_eV * m  # in unit of 1/m
t0_tau1 = 0 # -0.25


t_i = -10 * tau1
t_f = 10 * tau1
N = 20001
ts = np.linspace(t_i, t_f, N)


def single_sauter_data_name(E0, tau_eV):
    return 'data/single_sauter/' + f'E0={E0:.2f}Ec_tau={tau_eV:.0e}eV.npz'


def double_sauter_data_name():
    return ('data/double_sauter/'
            + f'E1={E1:.2f}_t1={tau1_eV:.0e}eV+'
            + f'E2={E2:.3f}_t2={tau2_eV:.0e}eV+t0{t0_tau1:.2f}t1.npz')


def compute_single_sauter(recompute=False):
    E0 = 0.25
    tau_eV = 1e-4  # in unit of eV
    tau = tau_eV * m  # in unit of 1/m

    data_filename = single_sauter_data_name(E0, tau_eV)

    if not os.path.exists(data_filename) or recompute:
        # define E and A as Callable functions
        A_func, E_func = sauter_pulse(E0, tau)

        p_vals = np.linspace(-10, 10, 200)  # unit: m

        def psi_final(p):
            return evolve(ts, A_func, E_func, p=p, only_final=True)

        alpha_vals, beta_vals = np.array([psi_final(p) for p in tqdm(p_vals)]).T

        np.savez(data_filename,
                 tau=tau,
                 e=E0,
                 ps=p_vals,
                 alpha=alpha_vals,
                 beta=beta_vals
                 )

    else:
        data = np.load(data_filename)
        p_vals = data['ps']
        alpha_vals = data['alpha']
        beta_vals = data['beta']

    demo_p_dependence(p_vals, alpha_vals, beta_vals, item='phase')
    plt.show()


def double_sauter_enhancement(recompute=False):
    data_filename = double_sauter_data_name()
    return p_spectrum({
            'tau1': tau1,
            'e1': E1,
            'tau2': tau2,
            'e2': E2,
            't0': t0_tau1 * tau1
        }, recompute)


def plot_enhancement(img_filename=None):
    single_data_filename = single_sauter_data_name(E1, tau1_eV)
    double_data_filename = double_sauter_data_name()

    data_single = np.load(single_data_filename)
    data_double = np.load(double_data_filename)

    ps = data_single['ps']
    beta_single = data_single['beta']
    beta_double = data_double['beta']
    f_single = 2 * np.abs(beta_single) ** 2
    f_double = 2 * np.abs(beta_double) ** 2

    plt.plot(ps, f_single, '--', label='single pulse')
    plt.plot(ps, f_double, label='double pulse')
    plt.xlabel('p/m')
    plt.ylabel('f(p)')
    plt.legend()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.tight_layout()
    if img_filename is not None:
        plt.savefig(img_filename, dpi=400)
    plt.show()


if __name__ == '__main__':
    # strengthen(recompute=False)
    # compute_single_sauter(recompute=False)
    # double_sauter_enhancement()
    # plot_enhancement(img_filename='figures/double_sauter/interference.pdf')
    print(tau2)