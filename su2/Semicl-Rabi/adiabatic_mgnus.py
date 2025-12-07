import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ellipeinc

from exact_solution import quasi_energy

plt.rcParams['font.size'] = 14

import yaml


def g(t, a, d):
    m = np.sqrt(a ** 2 + d ** 2)
    k = a / m
    return ellipeinc(t, k ** 2)


def phi(t, a, d):
    m = np.sqrt(a ** 2 + d ** 2)
    return m * g(t, a, d)


def v(t, a, d):
    m = np.sqrt(a ** 2 + d ** 2)
    k = a / m

    cos = np.sin(t)
    gamma = k * np.sqrt(1 - k ** 2) * cos / (1 - k ** 2 * cos ** 2) / 2
    exp_i_phi = np.exp(1j * m * g(t, a, d))
    return gamma * exp_i_phi


def insert_nans_at_jumps(x, y, jump_size=0.5):
    """
    Insert NaN values into x and y arrays at points where a large jump occurs,
    using vectorized NumPy operations (no Python loops).
    """
    dy = np.abs(np.diff(y))

    # Indices where jump occurs (i.e., |Δy| > threshold)
    jump_indices = np.where(dy > jump_size)[0]

    if jump_indices.size == 0:  # No jumps found
        return x, y

    # Compute insertion positions (after each jump index)
    insert_positions = jump_indices + 1

    # Insert NaNs into x and y arrays at the detected positions
    x_out = np.insert(x, insert_positions, np.nan)
    y_out = np.insert(y, insert_positions, np.nan)
    return x_out, y_out


def adiabatic_limit():
    a = 10
    d_vals = np.linspace(10.65, 10.7, 400)

    eps_exact_vals = np.array([quasi_energy(d, d, real_only=True) for d in d_vals])

    phi_vals = phi(np.pi, d_vals, d_vals) / (2 * np.pi)  # at T/2
    eps_adiabatic_vals = (phi_vals + 1 / 2) % 1 - 1 / 2

    line1 = plt.plot(d_vals, eps_exact_vals, label='Exact')
    color1 = line1[0].get_color()
    plt.plot(d_vals, -eps_exact_vals, color=color1)

    x, y = insert_nans_at_jumps(d_vals, eps_adiabatic_vals)
    plt.plot(x, y, '--', label='Adiabatic Approx')
    # plt.title(r'Quasienergy along $\Delta=A$')
    plt.xlabel(r'$\Delta=A$')
    plt.ylabel(r'$\epsilon$')
    # plt.grid()

    plt.legend()
    plt.xlim(10.67, 10.70)
    plt.ylim(0.4975, 0.5)
    plt.tight_layout()
    plt.savefig('figures/limiting/eps_adiabatic_limit.png', dpi=400)
    plt.show()


if __name__ == '__main__':
    adiabatic_limit()
    # print(phi(np.pi, 1, 1) / 2 / np.pi)
