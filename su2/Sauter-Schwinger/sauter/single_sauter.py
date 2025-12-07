from su2.common.adiabatic.integrator import evolve

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.constants import m_e, eV, c

from sauter_pulse import sauter_pulse, demo_p_dependence

import os

m = m_e * c ** 2 / eV

tau1 = 20
e1 = 1  # field strength, in unit of E_c


# define E and A as Callable functions
A_func, E_func = sauter_pulse(e1, tau1)


def time_evo(p=0, recompute=False):
    """
    Compute p-dependence of pair production rate for double Sauter pulse.
    """
    p = -40

    data_filename = ('data/single_sauter/' + f'evol_E0={e1:.2f}_tau={tau1}_p={p:.1f}.npz')
    if os.path.exists(data_filename) and not recompute:
        data = np.load(data_filename)
        ts = data['ts']
        alpha_vals = data['alpha']
        beta_vals = data['beta']
    else:
        # time range for evolution, taken sufficiently large
        t_i = -30 * tau1
        t_f = 30 * tau1
        N = 60001
        ts = np.linspace(t_i, t_f, N)

        data = evolve(ts, A_func, E_func, p)
        alpha_vals, beta_vals = data['psi_t'].T  # final state = [alpha, beta]

        np.savez(data_filename,
                 ts=ts,
                 tau=tau1,
                 e=e1,
                 p=p,
                 alpha=alpha_vals,
                 beta=beta_vals)

    mask = (ts >= 25 * tau1) & (ts <= 30 * tau1)
    ts_demo = ts[mask] / tau1  # in unit of tau1
    plt.plot(ts_demo, beta_vals.real[mask], label='Re')
    plt.plot(ts_demo, beta_vals.imag[mask], label='Im')

    plt.xlabel(r'$t/\tau$')
    plt.ylabel(r'$\beta(t)$')
    plt.title(f'Time Evolution of β(t) at p={p:.1f}, E0={e1:.2f}, τ={tau1}')
    plt.legend()

    # plt.axvline(x=0, linestyle='--', color='gray', alpha=0.7)
    plt.grid(True)
    plt.xlim(ts_demo[0], ts_demo[-1])
    plt.tight_layout()
    plt.show()

    phase = np.angle(beta_vals/alpha_vals)
    plt.plot(ts_demo, phase[mask], label='phase')
    plt.show()

    from su2.common.bloch import make_movie

    psi = np.array([alpha_vals[mask], beta_vals[mask]]).T
    # make_movie(psi, filename=f'figures/single_sauter/evol_E0={e1:.2f}_tau={tau1}_p={p:.1f}.gif')


def final_p_dependence(recompute=False, item='beta'):
    """
    Compute time evolution of beta(t) for double Sauter pulse.
    """

    # double Sauter pulse parameters, NOTE: change filename accordingly!

    data_filename = ('data/single_sauter/' + 'final_p_dep.npz')

    def get_final_state(p):
        data = evolve(ts, A_func, E_func, p=p)
        psi_final = data['psi_t'][-1]  # final state = [alpha, beta]
        alpha, beta = psi_final
        return alpha, beta

    if not os.path.exists(data_filename) or recompute:
        # time range for evolution, taken sufficiently large
        t_i = -10
        t_f = 10
        N = 20001
        ts = np.linspace(t_i, t_f, N) * tau1

        ps = np.linspace(0, 1, 100)  # unit: m
        alpha_vals, beta_vals = np.array([get_final_state(p) for p in tqdm(ps)]).T

        np.savez(data_filename,
                 tau=tau1,
                 e=e1,
                 ps=ps,
                 alpha=alpha_vals,
                 beta=beta_vals
                 )
    else:
        data = np.load(data_filename)

        ps = data['ps']
        alpha_vals = data['alpha']
        beta_vals = data['beta']

    demo_p_dependence(ps, alpha_vals, beta_vals, item=item)


if __name__ == '__main__':
    # final_p_dependence(recompute=True, item='phase')
    time_evo(recompute=True)