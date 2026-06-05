import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import os

plt.rcParams['font.size'] = 14


def delta_theta(p, A0):
    return np.arctan(p) - np.arctan(p - A0)


def E_p(p):
    return np.sqrt(1 + p ** 2)


def IR_r_v(r, A0, v, p_max=400.0, N=20001):
    """
    Numerical evaluation of
    I(r) = ∫_{-∞}^{∞} (1/(2π)) * ( sin(Δθ_p) / (2 E_p) ) * [ e^{i(p+E_p/v) r} + e^{i(p-E_p/v) r} ] dp
    using trapezoidal rule on [-p_max, p_max] with N points.
    """
    p = np.linspace(-p_max, p_max, N)
    dtheta = delta_theta(p, A0)
    pref = (np.sin(dtheta) / (2.0 * E_p(p))) / (2.0 * np.pi)  # shape (N,)
    phase1 = np.exp(1j * (p + E_p(p) / v) * r)
    phase2 = np.exp(1j * (p - E_p(p) / v) * r)
    integrand = pref * (phase1 + phase2)
    return trapezoid(integrand, p)


def IR_t_r(t, r, A0, p_max=400.0, N=20001):
    """
    Numerical evaluation of
    I(r) = ∫_{-∞}^{∞} (1/(2π)) * ( sin(Δθ_p) / (2 E_p) ) * [ e^{i(p+E_p/v) r} + e^{i(p-E_p/v) r} ] dp
    using trapezoidal rule on [-p_max, p_max] with N points.
    """
    p = np.linspace(-p_max, p_max, N)
    dtheta = delta_theta(p, A0)

    pref = (np.sin(dtheta) / (2.0 * E_p(p))) / (2.0 * np.pi)  # shape (N,)
    phase1 = np.exp(1j * (p * r + E_p(p) * 2 * t))
    phase2 = np.exp(1j * (p * r - E_p(p) * 2 * t))
    integrand = pref * (phase1 + phase2)
    return trapezoid(integrand, p)


def IR_t_r_derivative(t, r, A0, p_max=400.0, N=20001):
    """
    -i ∂I_R/∂r
    """
    p = np.linspace(-p_max, p_max, N)
    dtheta = delta_theta(p, A0)

    pref = p * (np.sin(dtheta) / (2.0 * E_p(p))) / (2.0 * np.pi)  # shape (N,)
    phase1 = np.exp(1j * (p * r + E_p(p) * 2 * t))
    phase2 = np.exp(1j * (p * r - E_p(p) * 2 * t))
    integrand = pref * (phase1 + phase2)
    I_val = trapezoid(integrand, p)
    return I_val

# ------------------ asymptotic (0 < v < 1) ----------------------------
def II_t_r(t, r, A0, p_max=400.0, N=20001):
    """
    I_I(r) = ∫ (dp / 2π) * ( sin(Δθ_p) / (2i) ) * [ e^{i(p+E_p/v)r} - e^{i(p-E_p/v)r} ]
    """
    p = np.linspace(-p_max, p_max, N)
    dtheta = delta_theta(p, A0)
    pref = (np.sin(dtheta) / 2j) / (2 * np.pi)
    phase1 = np.exp(1j * (p * r + E_p(p) * 2 * t))
    phase2 = np.exp(1j * (p * r - E_p(p) * 2 * t))
    integrand = pref * (phase1 - phase2)
    return trapezoid(integrand, p)


def I_I(r, A0, v, p_max=400.0, N=200001):
    """
    I_I(r) = ∫ (dp / 2π) * ( sin(Δθ_p) / (2i) ) * [ e^{i(p+E_p/v)r} - e^{i(p-E_p/v)r} ]
    """
    p = np.linspace(-p_max, p_max, N)
    dtheta = delta_theta(p, A0)
    pref = (np.sin(dtheta) / 2j) / (2 * np.pi)
    phase1 = np.exp(1j * (p + E_p(p) / v) * r)
    phase2 = np.exp(1j * (p - E_p(p) / v) * r)
    integrand = pref * (phase1 - phase2)
    return trapezoid(integrand, p)


def I_beta(r, A0, p_max=200.0, N=40000):
    def integrand(p):
        delta = delta_theta(p, A0)
        _beta2_e = (2 * np.sin(delta / 2) ** 2) / E_p(p) / (2 * np.pi)
        return _beta2_e * np.exp(1j * p * r)

    p_vals = np.linspace(-p_max, p_max, N)
    f_vals = integrand(p_vals)
    return trapezoid(f_vals, p_vals)


def I_beta_derivative(r, A0, p_max=200.0, N=40000):
    def integrand(p):
        delta = delta_theta(p, A0)
        _beta2_e = p * (2 * np.sin(delta / 2) ** 2) / E_p(p) / (2 * np.pi)
        return _beta2_e * np.exp(1j * p * r)

    p_vals = np.linspace(-p_max, p_max, N)
    f_vals = integrand(p_vals)
    return trapezoid(f_vals, p_vals)


def I_R_asy(r, A0, v):
    """
    Asymptotic formula for 0 < v < 1 derived earlier:
      I(r) ~ (A0 * sqrt(v*kappa) / (2*sqrt(2*pi*r))) * [ e^{ i( r*kappa/v + pi/4 ) }/E_{p_+ - A0}
                                                         + e^{-i( r*kappa/v + pi/4 )}/E_{p_- - A0} ]
    with p_+ = -v/kappa, p_- = +v/kappa, kappa = sqrt(1-v^2).
    """
    if not (0 < v < 1):
        return np.nan
    kappa = np.sqrt(1.0 - v ** 2)
    p_plus = -v / kappa  # p_+  (note sign convention used in derivation)
    p_minus = +v / kappa  # p_-
    E_p_plus_A0 = E_p(p_plus - A0)
    E_p_minus_A0 = E_p(p_minus - A0)
    pref = A0 * np.sqrt(v * kappa) / (2.0 * np.sqrt(2.0 * np.pi * r))
    phase = r * kappa / v + np.pi / 4.0
    return pref * (np.exp(1j * phase) / E_p_plus_A0 + np.exp(-1j * phase) / E_p_minus_A0)


def demo_IR():
    # ------------------ parameters and compute ----------------------------
    A0 = 0.8
    v = 0.2  # must satisfy 0 < v < 1 for asymptotic
    p_max = 200.0
    N = 100001  # odd number gives symmetric node at 0

    r_vals = np.linspace(.1, 40.0, 400)  # sample r values (start where asymptotics become relevant)

    I_num = np.array([IR_r_v(r, A0, v, p_max=p_max, N=N) for r in r_vals])
    I_asym = np.array([I_R_asy(r, A0, v) for r in r_vals])

    # ------------------ plotting ------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(r_vals, I_num.real, label=r"numerical")
    plt.plot(r_vals, I_asym.real, '--', label=r"asymptotic")
    plt.xlabel(r"$r$")
    plt.ylabel(r"Re{$I(r)$}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Optional: also plot imaginary part and absolute value for completeness
    plt.figure(figsize=(8, 4))
    plt.plot(r_vals, I_num.imag, label=r"numerical)")
    plt.plot(r_vals, I_asym.imag, '--', label=r"asymptotic")
    plt.xlabel(r"$r$")
    plt.ylabel(r"Im{$I(r)$}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ------------------ asymptotic from pole p = i ------------------------


def I_beta_pole_asymptotic(r, A0):
    """
    Leading asymptotic from pole at p = i:
      I_beta(r) ~ C * exp(-r),  where C = i * A0 / (2 * E_{i - A0}).
    """
    E_iA0 = np.sqrt(1.0 + (1j - A0) ** 2)  # E_{i - A0}
    C = 1j * A0 / (2.0 * E_iA0)
    return C * np.exp(-r)


def demo_beta():
    A0 = 0.6
    data_filename = f'data/I_beta_asymptotic_A0={A0:.1f}.npz'

    if not os.path.exists('figures'):
        os.makedirs('figures')

    if os.path.exists(data_filename):
        data = np.load(data_filename)
        A0 = data['A0'].item()
        r_vals = data['r']
        I_num = data['I_num']
        I_pole = data['I_pole']
    else:
        p_max = 400.0
        N = 200001  # large, odd -> symmetric node at 0
        r_vals = np.linspace(0, 40.0, 200)  # r range where asymptotic is expected to apply
        # numerical integration (may take some time for large N)
        I_num = np.array([I_beta(r, A0, p_max=p_max, N=N) for r in r_vals])
        I_pole = np.array([I_beta_pole_asymptotic(r, A0) for r in r_vals])

        np.savez(data_filename, A0=A0, r=r_vals, I_num=I_num, I_pole=I_pole)

    plt.figure(figsize=(8, 4))
    plt.plot(r_vals, np.abs(I_num), label=r"$|I_\beta(r)|$ (numeric)")
    plt.plot(r_vals, np.abs(I_pole), '--', label=r"$|C|e^{-r}$ (pole asymptotic)")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$|I_\beta(r)|$")
    # plt.title(r"Asymptotic behavior of $I_\beta(r)$ from pole at $p=i$", fontsize=13)
    plt.legend()
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'figures/I_beta_asymptotic_A={A0:.1f}.pdf', dpi=400)
    plt.show()


# numerical integral of I_I


# asymptotic stationary-phase approximation for 0<v<1
def I_I_asymptotic(r, A0, v):
    kappa = np.sqrt(1 - v ** 2)
    p_plus = v / kappa
    p_minus = -v / kappa
    E_plus = np.sqrt(1 + (p_plus - A0) ** 2)
    E_minus = np.sqrt(1 + (p_minus - A0) ** 2)
    amp = (A0 / 1j) * np.sqrt(np.pi * v / (2 * r)) * kappa ** (-0.5) / (2 * np.pi)
    phase = r * kappa / v + np.pi / 4
    return amp * (np.exp(1j * phase) / E_minus - np.exp(-1j * phase) / E_plus)


def demo_II():
    A0 = 0.6
    v = 0.6
    r_vals = np.linspace(0.1, 25, 400)
    t_vals = r_vals / (2 * v)
    I_num = np.array([II_t_r(t, r, A0) for t, r in zip(t_vals, r_vals)])
    I_asym = np.array([I_I_asymptotic(r, A0, v) for r in r_vals])

    # ------------------ plotting ------------------------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(r_vals, I_num.real, label='Re $I_I(r)$ (numeric)')
    plt.plot(r_vals, I_asym.real, '--', label='Re asymptotic')
    plt.xlabel(r"$r$")
    plt.ylabel(r"Re $I_I(r)$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(r_vals, np.abs(I_num), label=r"$|I_I(r)|$ (numeric)")
    plt.plot(r_vals, np.abs(I_asym), '--', label=r"$|I_I(r)|$ asymptotic")
    plt.xlabel(r"$r$")
    plt.ylabel(r"$|I_I(r)|$")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'figures/I_I_asymptotic_A={A0:.1f}_v={v:.1f}.pdf', dpi=400)
    plt.show()


def IR_asy_derivative(r, A0, t):
    v = r / (2 * t)
    s = np.sqrt(4 * t ** 2 - r ** 2) / 2
    kappa = s / t
    p_plus = v / kappa
    p_minus = -v / kappa
    E_plus = np.sqrt(1 + (p_plus - A0) ** 2)
    E_minus = np.sqrt(1 + (p_minus - A0) ** 2)
    # 渐近公式: -i dI_R/dr
    return (A0 * v / 4) * np.sqrt(1 / (np.pi * s)) * (
                np.exp(1j * (2 * s + np.pi / 4)) / E_plus - np.exp(-1j * (2 * s + np.pi / 4)) / E_minus)


# 演示函数
def IR_derivative_demo(A0, t, r_min=0.1, r_max=None, N_r=500):
    if r_max is None:
        r_max = 2 * t * 0.99
    r_vals = np.linspace(r_min, r_max, N_r)
    dr = r_vals[1] - r_vals[0]

    # 数值差分计算 dI_R/dr
    I_vals = np.array([IR_r_v(r, A0, v=r / (2 * t)) for r in r_vals])
    dI_num = np.diff(I_vals) / dr
    r_num = r_vals[:-1] + dr / 2  # 对应差分位置

    # 渐近导数
    dI_asym = np.array([IR_asy_derivative(r, A0, t) for r in r_num])

    # 绘图比较
    plt.figure(figsize=(8, 4))
    plt.plot(r_num, -1j * dI_num.real, 'b', label='-i ∂r I_R (num diff, real)')
    plt.plot(r_num, -1j * dI_num.imag, 'b--', label='-i ∂r I_R (num diff, imag)')
    plt.plot(r_num, dI_asym.real, 'r', label='Asymptotic real')
    plt.plot(r_num, dI_asym.imag, 'r--', label='Asymptotic imag')
    plt.xlabel('r')
    plt.ylabel('-i ∂r I_R')
    plt.legend()
    plt.grid(True)
    plt.show()

    return r_num, dI_num, dI_asym


if __name__ == '__main__':
    demo_beta()
    # demo_IR()
    # demo_II()
    # IR_derivative_demo(A0=0.8, t=10.0)