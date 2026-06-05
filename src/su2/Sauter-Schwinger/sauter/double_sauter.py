import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from sauter_pulse import demo_p_dependence
from solver import DoubleSauter

"""

"""

from params import e1, e2, tau1, tau2, t0_tau1

plt.rcParams['font.size'] = 14

t_i = -10 * tau1
t_f = 10 * tau1
N = 20001
ts = np.linspace(t_i, t_f, N)


def double_sauter_data_name():
    if e2 == 0:
        return 'data/single_sauter/' + f'E1={e1:.2f}_t1={tau1:.0e}.npz'
    else:
        return ('data/double_sauter/'
                + f'E1={e1:.2f}_t1={tau1:.0e}+'
                + f'E2={e2:.3f}_t2={tau2:.0e}+t0{t0_tau1:.2f}t1.npz')


def p_spectrum(params, data_filename, recompute=False):
    if not os.path.exists(data_filename) or recompute:
        # define E and A as Callable functions
        solver = DoubleSauter(params['e1'], params['tau1'],
                              params['e2'], params['tau2'],
                              t0=params['t0'])

        p_vals = np.linspace(0, 3, 300)  # unit: m

        alpha_vals, beta_vals = np.array([solver.psi_final(p) for p in tqdm(p_vals)]).T
        np.savez(data_filename,
                 tau1=tau1,
                 e1=e1,
                 tau2=tau2,
                 e2=e2,
                 t0=t0_tau1,
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
    plt.tight_layout()
    plt.show()


def plot_enhancement(img_filename=None):
    single_data_filename = double_sauter_data_name()
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
    data_fn = double_sauter_data_name()
    p_spectrum({
        'tau1': tau1,
        'e1': e1,
        'tau2': tau2,
        'e2': e2,
        't0': t0_tau1 * tau1
    }, data_fn, recompute=True)
    # plot_enhancement(img_filename='figures/double_sauter/interference.pdf')
