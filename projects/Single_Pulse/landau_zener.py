import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid


from adiabatic_picture import h_num, v_func_num
from su2.common.su2_integrator import make_h_interp, evolve


def f_lz(t, v):
    return v * t


def f_dot_lz(t, v):
    return v


def demo():
    v = 1
    d = 1

    g = d**2 / (4 * v)
    p_exact = np.exp(-2 * np.pi * g)

    ts = np.linspace(-400, 400, 20000) / np.sqrt(v)

    h_vals = h_num(ts, f_lz, f_dot_lz, d, v)
    print('h_vals computed')

    h_func = make_h_interp(ts, h_vals)
    print('h_func created')

    v_num = v_func_num(ts, f_lz, f_dot_lz, d, v)

    psi_i = np.array([1, 0])
    # s = u(ts[-1], ts[0], len(ts), h_func)
    psi_ts = evolve(psi_i, ts[-1], ts[0], len(ts), h_func)
    alpha, beta = psi_ts.T
    psi_f = psi_ts[-1]

    p_num = 1 - np.abs(psi_f[0])**2

    print(p_num, p_exact)

    beta_1st = cumulative_trapezoid(v_num, ts, initial=0)

    p_1st = np.abs(beta_1st[-1])**2
    print(p_1st)

    plt.plot(ts * np.sqrt(v), beta.real, label='Re')
    plt.plot(ts * np.sqrt(v), beta.imag, label='Im')
    plt.show()


if __name__ == '__main__':
    demo()
