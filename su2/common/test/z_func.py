import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

from su2.common.test.root import find_roots

mp.mp.dps = 50  # Set high precision for mpmath (50 decimal places)


def theta(t):
    """Riemann–Siegel theta function"""
    return mp.im(mp.log(mp.gamma(0.25 + 0.5j * t))) - t * mp.log(mp.pi) / 2


def Z(t):
    """Compute Z(t) = e^{iθ(t)} ζ(1/2 + it) (real-valued on the real axis)"""
    val = mp.exp(1j * theta(t)) * mp.zeta(0.5 + 1j * t)
    assert abs(mp.im(val)) < 1e-40, f"Imaginary part too large: {mp.im(val)}"
    return mp.re(val)


def Z_siegel(t):
    """
    Approximate Z(t) using the Riemann–Siegel formula without remainder term.
    N = floor(sqrt(t / (2π))) is the truncation point.
    """
    N = int(mp.floor(mp.sqrt(t / (2 * mp.pi))))
    th = theta(t)
    # Sum over n from 1 to N: cos(t log n - θ(t)) / sqrt(n)
    s = mp.nsum(lambda n: mp.cos(t * mp.log(n) - th) / mp.sqrt(n), [1, N])
    return 2 * s  # Multiply by 2 as per the Riemann–Siegel formula


def demo_riemann_siegel():
    # Define a range of t values to plot
    t_values = np.linspace(0, 10, 200) + 50

    # Compute exact Z(t) using ζ and approximate values using the Riemann–Siegel formula
    zeta_vals = [Z(t) for t in t_values]
    siegel_vals = [Z_siegel(t) for t in t_values]

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(t_values, zeta_vals, label='Exact Z(t) from ζ', lw=2)
    plt.plot(t_values, siegel_vals, '--', label='Riemann–Siegel Approximation', lw=1.5)
    plt.title("Riemann–Siegel Formula vs Exact Zeta Function")
    plt.xlabel("t")
    plt.ylabel("Z(t)")
    plt.legend()
    plt.grid(True)
    plt.savefig('figures/riemann_siegel.pdf', dpi=400)
    # plt.show()


def plot_z():
    t_values = np.linspace(0, 42, 400)
    def z_func(t): return float(Z(t))
    zeros, zeta_vals = find_roots(t_values, z_func)

    print("First few non-trivial zeros of the Riemann zeta function:")
    for i, z in enumerate(zeros):
        print(i+1, z)

    plt.figure(figsize=(10, 5))
    plt.plot(t_values, zeta_vals, label='Z(t)', lw=2)

    # Mark zeros
    plt.plot(zeros, np.zeros_like(zeros), 'rx', label='Zeros', markersize=5)
    for z in zeros:
        plt.text(z, 0.1, f'{z:.4f}', fontsize=8, ha='center')

    # plt.title("Riemann Z Function Z(t)")
    plt.xlabel("t")
    plt.ylabel("Z(t)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/riemann_z.pdf', dpi=400)
    # plt.show()


if __name__ == '__main__':
    # plot_z()
    demo_riemann_siegel()
    # print(type(Z(1000)))
