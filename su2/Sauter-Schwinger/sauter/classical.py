import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from tqdm import tqdm

from exact_solution import A


def velocity(t, p, e, tau):
    """ integrand for classical correlation function calculation """
    p_kin = p + A(t, e, tau)
    E_p = np.sqrt(1 + p_kin ** 2)
    return p_kin / E_p


def demo():
    E0 = 10
    tau = 0.5

    ps = np.linspace(-10, 10, 80)

    t_vals = np.linspace(0, 5 * tau, 1000)

    for p in tqdm(ps):
        v_vals = velocity(t_vals, p, E0, tau)
        r_vals = cumulative_trapezoid(v_vals, t_vals, initial=0)

        plt.plot(r_vals, t_vals, color='k', linewidth=0.5)

    plt.plot(t_vals, t_vals, '--', label='light cone |r|=t', color='gray')
    plt.plot(-t_vals, t_vals, '--', color='gray')

    plt.xlabel('Position r(t)')
    plt.ylabel('Time t')
    plt.legend()
    plt.savefig('figures/classical_trajectories.pdf', dpi=400)


if __name__ == '__main__':
    demo()
