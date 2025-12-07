import numpy as np
import matplotlib.pyplot as plt


"""
Reproduce plots in He, R., Ai, MZ., Cui, JM. et al. 
Riemann zeros from Floquet engineering a trapped-ion qubit. 
npj Quantum Inf 7, 109 (2021).

Last updated: 09-25 2025
"""

# Parameters
E = 1.0
t_min, t_max = 0.0, 2 * np.pi
N = 100
t = np.linspace(t_min, t_max, N)
dt = np.mean(np.diff(t))

plt.rcParams['font.size'] = 18


def tilde_g(t):
    return np.exp(t/2) - np.exp(-t/2) * np.floor(np.exp(t))


def capital_f(t):
    return np.arccos(- tilde_g(t) * np.cos(E * t))


def plot_capital_f():
    F = capital_f(t)
    plt.plot(t, F, lw=1, label=r"$F(t)$")
    plt.xlabel("t")
    plt.ylabel("F(t)")
    plt.xlim(t_min, t_max)
    plt.title("Capital F(t), E=1")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()


def plot_f():
    F = capital_f(t)
    f = np.gradient(F, dt)
    plt.plot(t, f, lw=1, label=r"$f(t)=\frac{dF}{dt}$")
    plt.xlabel("t")
    plt.ylabel("f(t)")
    plt.xlim(t_min, t_max)
    plt.title("f(t), E=1")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()


def fourier_series(f, t, N_terms):
    """
    Compute the Fourier series coefficients (a0, a_n, b_n) and
    the truncated Fourier series approximation of f(t).

    Parameters
    ----------
    f : array_like
        Function values at points t.
    t : array_like
        Sample points over [0, 2*pi].
    N_terms : int
        Number of Fourier terms to keep.

    Returns
    -------
    a0 : float
        Constant term.
    a_n : ndarray
        Cosine coefficients, length N_terms.
    b_n : ndarray
        Sine coefficients, length N_terms.
    f_truncated : ndarray
        Truncated Fourier series approximation of f at points t.
    """
    dt = t[1] - t[0]
    a0 = np.sum(f) * dt / (2 * np.pi)

    a_n = np.zeros(N_terms)
    b_n = np.zeros(N_terms)
    for n in range(1, N_terms + 1):
        a_n[n - 1] = np.sum(f * np.cos(n * t)) * dt / np.pi
        b_n[n - 1] = np.sum(f * np.sin(n * t)) * dt / np.pi

    # Reconstruct truncated Fourier series
    f_truncated = np.full_like(f, a0)
    for n in range(1, N_terms + 1):
        f_truncated += a_n[n - 1] * np.cos(n * t) + b_n[n - 1] * np.sin(n * t)

    return a0, a_n, b_n, f_truncated


def derivative_from_coeffs(a0, a_n, b_n, t):
    """
    Compute derivative f'(t) from Fourier coefficients of
    f(t) = a0 + sum_{n=1}^N (a_n cos(nt) + b_n sin(nt)).

    Parameters
    ----------
    a0 : float
        Constant term (unused for derivative).
    a_n : array_like, shape (N_terms,)
        Cosine coefficients a_1 ... a_N.
    b_n : array_like, shape (N_terms,)
        Sine coefficients b_1 ... b_N.
    t : array_like, shape (M,)
        Points where to evaluate derivative.

    Returns
    -------
    fprime : ndarray, shape (M,)
        Values of f'(t) at the points t.
    a_n_prime : ndarray, shape (N_terms,)
        Cosine coefficients of f'(t): a_n' = n * b_n.
    b_n_prime : ndarray, shape (N_terms,)
        Sine coefficients of f'(t): b_n' = -n * a_n.
    """
    a_n = np.asarray(a_n)
    b_n = np.asarray(b_n)
    t = np.asarray(t)
    N_terms = a_n.size
    n = np.arange(1, N_terms + 1)  # shape (N_terms,)

    # derivative coefficients
    a_n_prime = n * b_n
    b_n_prime = -n * a_n
    # a0' = 0 (not returned explicitly)

    # evaluate f'(t) vectorized:
    # compute phases shape: (N_terms, M)
    nt = n[:, None] * t[None, :]  # shape (N_terms, M)
    # sum over n: -n a_n sin(nt) + n b_n cos(nt)
    term = (- (n * a_n)[:, None] * np.sin(nt)) + ((n * b_n)[:, None] * np.cos(nt))
    fprime = np.sum(term, axis=0)

    return fprime, a_n_prime, b_n_prime


def demo_fourier():
    N_points = 5000
    t = np.linspace(0, 2 * np.pi, N_points, endpoint=False)
    N_terms = 500

    # Define a function
    f = capital_f(t)
    f = np.concatenate((f[::2], -f[::-1][::2]))

    # Compute Fourier series
    a0, a_n, b_n, f_truncated = fourier_series(f, t, N_terms)

    # # Plot original and truncated series
    # plt.figure(figsize=(10, 6))
    # plt.plot(t, f, label='Original', lw=1)
    # plt.plot(t, f_truncated, label=f'Truncated(N={N_terms})', lw=1, linestyle='--')
    # plt.xlim(0, 2 * np.pi)
    # plt.xlabel('t')
    # plt.ylabel('f(t)')
    # plt.show()

    # Compute derivative from Fourier coefficients
    df = np.gradient(f, t[1] - t[0])
    df_truncated, a_n_prime, b_n_prime = derivative_from_coeffs(a0, a_n, b_n, t)

    plt.figure(figsize=(10, 4))
    plt.plot(2*t/np.pi, df, label='Original Derivative', lw=1)
    # plt.plot(2*t, df_truncated, label=f'Truncated Derivative(N={N_terms})', lw=1)
    plt.xlim(0, 4)
    plt.ylim(-10, 10)
    plt.xlabel(r'$t/\pi$')
    plt.ylabel(r"$\tilde{f}$ (t;E=1)")
    # plt.legend()
    plt.tight_layout()
    plt.savefig('figures/su2/driving_field.pdf', dpi=400)
    plt.show()


if __name__ == '__main__':
    # plot_capital_f()
    # plot_f()
    demo_fourier()
