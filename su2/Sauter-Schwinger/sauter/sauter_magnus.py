import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from params import A_func, E_func, tau1, e1, tau2, e2, t0_tau1
from sauter_pulse import demo_p_dependence
from solver import SauterSchwingerSolver, DoubleSauter
from su2.common.adiabatic.adiabatic_picture import su2_exp
from su2.common.bloch import xy_trajectory

plt.rcParams['font.size'] = 14


def p_spectrum_data_name(single_pulse=False):
    if np.abs(e2) <= 1e-15 or single_pulse:
        return 'data/demo/' + f'E={e1:.2f}_tau={tau1:.0e}.npz'
    else:
        return ('data/demo/'
                + f'E1={e1:.2f}_tau1={tau1:.0e}+'
                + f'E2={e2:.3f}_tau2={tau2:.0e}+t0{t0_tau1:.2f}tau1.npz')


def evolution_data_name(p, single_pulse=False):
    if np.abs(e2) <= 1e-15 or single_pulse:
        return 'data/demo/' + f'E={e1:.2f}_tau={tau1:.0e}_p={p:.2f}.npz'
    else:
        return ('data/demo/'
                + f'E1={e1:.2f}_tau1={tau1:.0e}+'
                + f'E2={e2:.3f}_tau2={tau2:.0e}+t0{t0_tau1:.2f}tau1_p={p:.2f}.npz')


def beta_t(p=0.0, item='ab', img_filename=None):
    t_vals = np.linspace(-10, 0.3, 100000) * tau1

    solver = SauterSchwingerSolver(A_func, E_func)
    psi_tp = solver.wkb_psi_tp

    wkb_solution = psi_tp(t_vals, p)
    alpha, beta = wkb_solution['alpha'], wkb_solution['beta']
    eta_vals = wkb_solution['eta']

    a1_asymp = eta_vals / (2j * solver.omega_p(t_vals, p))  # asymptotic form
    _, beta_asymp = su2_exp(a1_asymp)

    # numeric_data = evolve(t_vals, A_func, E_func, p)
    # _, beta_exact = numeric_data['psi_t'].T
    #
    mask = (t_vals >= -0.3 * tau1) & (t_vals <= 0.3 * tau1)
    t_demo = t_vals[mask] / tau1

    print(f"Final f at p={p:.3f}: {2 * np.abs(beta[-1])**2:.6e}")

    if item == 'phase':
        plt.figure(figsize=(8, 6))
        # phase_diff = np.angle(beta_asymp[mask]/(beta[mask]))
        # plt.plot(t_demo, phase_diff, label='Phase Difference Re[β(t)] - Re[β_asymp(t)]')
        phase = np.angle(beta[mask])
        phase = np.mod(phase, 2 * np.pi)
        plt.plot(t_demo, phase/np.pi)
        # plt.legend()
        plt.xlabel(r'$t/\ta'
                   r'u$')
        plt.ylabel(r'$\phi(t)/\pi$')
        plt.xlim(min(t_demo), max(t_demo))
        plt.ylim(0, 2)
    elif item == 'xy':
        psi = np.array([alpha, beta]).T
        xs, ys = xy_trajectory(psi)

        plt.figure(figsize=(6, 6))
        plt.plot(xs, ys)
        plt.plot(0, 0, 'rx', label=r'$t=-\infty$')

        plt.xlabel('X')
        plt.ylabel('Y')
        # plt.title('Bloch Sphere XY Projection')
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')

        idx_ac = np.argmin(np.abs(t_vals))  # advoided crossing
        t_ac = t_vals[idx_ac]
        x_ac, y_ac = xs[idx_ac], ys[idx_ac]
        plt.plot(x_ac, y_ac, '*', label='t=0')

        a = 0.35
        plt.xlim(-a, a)
        plt.ylim(-a, a)
        plt.legend()

    elif item == 'beta':
        plt.figure(figsize=(8, 5))
        # line_im = plt.plot(t_vals[mask], beta.imag[mask], label='Im[β(t)]')
        line_re = plt.plot(t_demo, beta.real[mask], label='wkb')
        #
        # # color_im = line_im[0].get_color()
        color_re = line_re[0].get_color()
        #
        # # plt.plot(t_vals[mask], beta_asymp.imag[mask], '--', color=color_im, label='Im[β_asymp(t)]')
        plt.plot(t_demo, beta_asymp.real[mask], '--', color=color_re, label='osc')
        # plt.plot(t_demo, beta_exact.real[mask], ':', color=color_re, label='exact')

        plt.plot(t_demo, beta.real[mask] - beta_asymp.real[mask], label='diff')
        plt.xlabel(r'$t/\tau$')
        plt.ylabel('Re[β(t)]')
        plt.legend()

        # plt.title('Time Evolution of Re[β(t)]' + f' at p={p:.3f}')

    else:
        f = 2 * np.abs(beta) ** 2
        plt.figure(figsize=(8, 5))
        plt.plot(t_demo, f[mask], label='f(t)')
        plt.xlabel(r'$t/\tau$')
        plt.ylabel('f(t)')
        plt.title('Time Evolution of f(t)' + f' at p={p:.3f}')
        plt.legend()

    plt.tight_layout()
    plt.grid(True)

    if img_filename is not None:
        plt.savefig(img_filename, dpi=400)
    plt.show()


def p_dependence():
    data_filename = p_spectrum_data_name()

    solver = DoubleSauter(e1, tau1, e2, tau2, t0=t0_tau1 * tau1)

    ps = np.linspace(-4, 4, 400)

    a1_vals = [solver.psi_final(p, method='wkb')[1] for p in tqdm(ps)]

    alpha_1st, beta_1st = su2_exp(a1_vals)

    np.savez(data_filename,
             tau1=tau1,
             e1=e1,
             tau2=tau2,
             e2=e2,
             t0=t0_tau1,
             ps=ps,
             alpha=alpha_1st,
             beta=beta_1st
             )
    demo_p_dependence(ps, alpha_1st, beta_1st, item='rate')
    plt.yscale('log')
    plt.show()


def plot_enhancement(item='rate', img_filename=None):
    single_data_filename = p_spectrum_data_name(True)
    double_data_filename = p_spectrum_data_name()

    data_single = np.load(single_data_filename)
    data_double = np.load(double_data_filename)

    ps = data_single['ps']
    alpha_single = data_single['alpha']
    alpha_double = data_double['alpha']

    beta_single = data_single['beta']
    beta_double = data_double['beta']

    if item == 'rate':
        f_single = 2 * np.abs(beta_single) ** 2
        f_double = 2 * np.abs(beta_double) ** 2

        plt.plot(ps, f_single, '--', label='single pulse')
        plt.plot(ps, f_double, label='double pulse')
        plt.ylabel(r'2$|\beta_p|^2$')

    else:
        phase_single = np.angle(beta_single / alpha_single)
        phase_single = np.mod(phase_single, 2 * np.pi)
        phase_double = np.angle(beta_double / alpha_double)
        phase_double = np.mod(phase_double, 2*np.pi)
        plt.plot(ps, phase_single/np.pi, '--', label='single pulse')
        plt.plot(ps, phase_double/np.pi, label='double pulse')

        plt.ylim(0, 2)
        plt.ylabel(r'$\phi_p/\pi$')

    plt.xlim(min(ps), max(ps))
    plt.xlabel('p/m')

    plt.legend()
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.tight_layout()
    if img_filename is not None:
        plt.savefig(img_filename, dpi=400)
    plt.show()


def plot_evolution_diff(p):
    single_data_filename = evolution_data_name(p=p, single_pulse=True)
    double_data_filename = evolution_data_name(p=p, single_pulse=False)

    data_single = np.load(single_data_filename)
    data_double = np.load(double_data_filename)
    t_vals = data_single['t_vals']
    beta_single = data_single['beta_t']
    beta_double = data_double['beta_t']
    plt.figure(figsize=(8, 5))


if __name__ == '__main__':
    # test_phi()

    p_dependence()

    # plot_enhancement('rate', img_filename='figures/demo/param3_rate.pdf')

    # beta_t(p=0, item='phase', img_filename='figures/demo/param1_phase_t_p=0_double.pdf')
