# Compare derivative d/dy arg Gamma(i y) with approximations
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np

mp.mp.dps = 50

"""
Phase of Gamma(i y) = arg Gamma(i y)

"""


def deriv_arg_gamma(y):
    """
    Exact derivative of arg Gamma(i y) with respect to y
    """
    # derivative = Re[ psi(i y) ],
    # because d/dy arg = Im(d/dy ln Gamma) = Im(i psi) = Re psi
    z = 1j * y
    psi = mp.digamma(z)
    return float(mp.re(psi))


def deriv_stirling(y):
    # derivative of y ln y - y - pi/4 is ln y
    return np.log(y) + 1 / (12 * y ** 2) - 1 / (360 * y ** 4)  # add some correction terms


def deriv_small_y(y, order=2):
    """
    Small-y expansion of derivative of arg Gamma(i y)
    theta'(y) = Re psi(i y) ~ -gamma + zeta(3) y^2 - zeta(5) y^4 + ...
    """
    gamma_const = float(mp.euler)
    zeta3 = float(mp.zeta(3))
    zeta5 = float(mp.zeta(5))

    val = -gamma_const
    if order >= 2:
        val += zeta3 * y ** 2
    if order >= 4:
        val -= zeta5 * y ** 4
    return val


def deriv_demo():
    # small-y approx: Re psi(i y) -> -gamma  as y->0
    gamma_const = float(mp.euler)  # = 0.5772...

    y_min = 0.1
    y_max = 5
    small_ys = np.linspace(y_min, 1, 100)
    large_ys = np.linspace(0.5, y_max, 100)
    ys = np.linspace(y_min, y_max, 100)
    exact = np.array([deriv_arg_gamma(y) for y in ys])
    stirling = np.array([deriv_stirling(y) for y in large_ys])
    small_approx = np.array([deriv_small_y(y, order=4) for y in small_ys])

    plt.figure(figsize=(8, 5))
    plt.plot(ys, exact, label="Exact derivative: Re ψ(i y)")
    plt.plot(large_ys, stirling, label="Stirling approx: ln y", linestyle="--")
    plt.plot(small_ys, small_approx, label=r"Small-y approx: $-\gamma$", linestyle=":")
    plt.xscale('symlog', linthresh=1e-2)  # show small region nicely
    plt.xlabel("y")
    plt.ylabel(r"$\partial_y \arg\Gamma(i y)$")
    plt.legend()
    plt.title("Derivative of phase: exact vs Stirling and small-y approximation")
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.tight_layout()
    plt.show()

    # # Also show relative error between exact and Stirling for y>1
    # mask = ys > 1
    # rel_err = (exact[mask] - stirling[mask]) / np.maximum(1e-12, np.abs(stirling[mask]))
    # plt.figure(figsize=(8, 4))
    # plt.plot(ys[mask], rel_err)
    # plt.axhline(0, color='k', lw=0.5)
    # plt.xlabel("y")
    # plt.ylabel("Relative error (exact - ln y)/|ln y|")
    # plt.title("Relative error of Stirling derivative vs exact (y>1)")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()


# exact arg Gamma(i y)
def arg_gamma_iy(y):
    return mp.arg(mp.gamma(1j * y))


# Stirling approx: arg Γ(i y) ≈ y ln y − y − π/4
def arg_gamma_stirling(y):
    return y * np.log(y) - y - np.pi / 4


def demo_phase():
    ys = np.linspace(0.001, 4, 600)
    arg_exact = [arg_gamma_iy(y) for y in ys]
    arg_exact = np.unwrap(arg_exact)  # unwrap phase
    arg_stirling = np.array([arg_gamma_stirling(y) for y in ys])

    plt.figure(figsize=(7, 5))
    plt.xlim(0, 4)
    plt.plot(ys, arg_exact / np.pi, label="arg Γ(i y) (exact)")
    plt.plot(ys, arg_stirling / np.pi, label="Stirling approx")
    plt.xlabel("y")
    plt.ylabel("phase")
    plt.legend()
    plt.title("Comparison: arg Γ(i y) vs Stirling approximation")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # demo_phase()
    deriv_demo()
