import numpy as np
from params import e1, e2, tau1, tau2
from solver import DoubleSauter
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
m = 1

plt.rcParams.update({'font.size': 14})


def g_complex(x):
    """
    2 int_0^x sqrt(1 + t^2) dt
    """
    x = x.astype(complex)
    return x * np.sqrt(1 + x ** 2) + np.arcsinh(x)


def phi_complex(t, p):
    x = (p + e1 * t) / m
    return m**2 / e1 * (g_complex(x) - g_complex(p/m))


# def phi_prime(x, a1, a2):
#     """Compute Φ'(x) = Im( z'(x) / z(x) )."""
#     w = x + 1j * c
#     S = np.sqrt(1 + w ** 2)
#     g_val = w * S + np.arcsinh(w)
#
#     a1 *= np.exp(-np.pi * m ** 2 / (2 * e1))
#
#     chi2 = np.imag(g_complex(x, c))
#     a2 *= np.exp(-np.abs(chi2) * m ** 2 / e1)
#
#     # z(x)
#     Ez = np.exp(-1j * g_val)
#     z = a1 + a2 * Ez
#
#     # g'(x) = 2 * sqrt(1 + (x+ic)^2) = 2*S
#     g_prime = 2 * S
#
#     # z'(x) = -i b g'(x) e^{-ig(x)}
#     z_prime = -1j * a2 * g_prime * Ez
#
#     # Φ'(x) = Im(z'(x)/z(x))
#     return np.imag(z_prime / z)


def beta_wkb_ddp(p, a1, a2):
    t_c1 = m / e1 * (-1j - p/m)
    t_c2 = - 1j * np.pi * tau2 / 2

    phi_c1 = phi_complex(t_c1, p)  # np.pi /2 * m**2 / e1
    phi_c2 = phi_complex(t_c2, p)
    assert np.all(phi_c2.imag < 0) and np.all(phi_c1.imag < 0)
    return a1 * np.exp(- 1j * phi_c1) + a2 * np.exp(- 1j * phi_c2)


def compare(item='rate'):
    a, b = 1.4, 0.15  # amplitudes from fitting

    ps = np.linspace(-4, 4, 400)
    beta_ddp = beta_wkb_ddp(ps, a, b)

    solver = DoubleSauter(e1, tau1, e2, tau2)
    solver_s = DoubleSauter(e1, tau1, 0, tau2)
    beta_vals = np.array([solver.psi_final(p, method='wkb')[1] for p in ps])
    beta_vals_s = np.array([solver_s.psi_final(p, method='wkb')[1] for p in ps])

    if item == 'rate':

        rates_wkb = np.abs(beta_vals) ** 2
        rates_ddp = np.abs(beta_ddp) ** 2

        fig, ax1 = plt.subplots()
        ax1.plot(ps, rates_wkb, label='Exact')
        ax1.plot(ps, rates_ddp, '--', label='Approx')
    elif item == 'beta':
        beta_vals *= -1
        fig, ax1 = plt.subplots()
        ax1.plot(ps, beta_vals.real, label='WKB Re')
        ax1.plot(ps, beta_ddp.real, '--', label='DDP Approx Re')

        ax1.plot(ps, beta_vals.imag, label='WKB Im')
        ax1.plot(ps, beta_ddp.imag, '--', label='DDP Approx Im')

    elif item == 'phase':
        plt.figure(figsize=(8, 5))

        phase_ddp = np.angle(beta_ddp)
        phase_ddp = np.unwrap(phase_ddp)
        dr_ddp = np.gradient(phase_ddp, ps)
        phase_wkb = np.angle(beta_vals)
        phase_wkb = np.unwrap(phase_wkb)
        dr_wkb = np.gradient(phase_wkb, ps)

        phase_wkb_s = np.angle(beta_vals_s)
        phase_wkb_s = np.unwrap(phase_wkb_s)
        dr_wkb_s = np.gradient(phase_wkb_s, ps)

        plt.plot(ps, dr_wkb, label='numeric')
        plt.plot(ps, dr_ddp, '--', label='complex WKB')
        plt.plot(ps, dr_wkb_s, label='single pulse', color='r', alpha=0.3)
        plt.axhline(2/e1, color='gray', linestyle='--', label=r'$\pm2m^2/E_0$')
        plt.axhline(- 2 / e1, color='gray', linestyle='--')
        plt.xlabel('p/m')
        plt.ylabel(r"$\phi'(p)$")
        # ax1.set_title(r'$E1=0.25E_c, E2=0.1E_1, \tau_1=10^{-4}eV^{-1}, \tau_2=0.02\tau_1$')

    plt.grid(True)
    # plt.xlabel('x')
    # plt.ylabel("$|R|^2$")
    # plt.yscale('log')
    plt.legend()
    plt.tight_layout()
    plt.savefig('offset_double_sauter.pdf', dpi=400)
    plt.show()


def check_chi1():
    def chi1(p):
        t_c1 = m / e1 * (1j - p/m)

        phi_c1 = phi_complex(t_c1, p)
        return phi_c1.imag

    ps = np.linspace(-4, 4, 400)
    chi1_vals = chi1(ps)
    plt.plot(ps, chi1_vals)
    plt.show()


def check_chi2():
    def chi2(p):
        t_c2 = - 1j * np.pi * tau2 / 2

        phi_c2 = phi_complex(t_c2, p)
        return phi_c2.imag

    ps = np.linspace(-4, 4, 800)
    chi2_vals = chi2(ps)
    plt.plot(ps, chi2_vals)
    plt.show()


def plot_J():
    def J_of_r(r):
        # numerical integration over y in [0, 20]
        y = np.linspace(0, 20, 20001)
        tanh_y = np.tanh(y)
        integrand = 1 / np.sqrt(r * r + 1) - tanh_y / np.sqrt(r * r + tanh_y * tanh_y)
        I = trapezoid(integrand, y)
        return I / r

    rs = np.linspace(0.001, 10, 1000)
    Js = np.array([J_of_r(r) for r in rs])

    plt.figure()
    plt.plot(rs, Js)
    plt.xlim(0, 10)
    plt.ylim(0, 1.1)
    plt.xlabel(r"$x=\frac{m}{A_0\tau}$")
    plt.ylabel("I(x)")
    plt.tight_layout()
    plt.savefig('figures/I_of_s.pdf', dpi=400)
    # plt.title("Numerical plot of J(r)")
    plt.show()


if __name__ == '__main__':
    # compare()
    # plot_J()
    # check_chi1()
    # check_chi2()
    compare('phase')
