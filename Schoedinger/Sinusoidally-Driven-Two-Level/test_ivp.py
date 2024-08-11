import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


t_span = (0, 1)
t_eval = np.linspace(*t_span, 10000)


def directly_solve():
    def h(t, y):
        return -1j * 2 * np.pi * y

    y0 = np.array([1.0], dtype=complex)

    # Solve the exact Schrödinger equation
    solution = solve_ivp(h, t_span, y0, t_eval=t_eval, method='RK45')
    return np.abs(solution.y[0]) ** 2
    # return np.real(solution.y[0])


def solve_by_part():
    def h(t, y):
        return 2 * np.pi * y[::-1] * np.array([1, -1])

    y0 = np.array([1.0, 0.0])

    # Solve the exact Schrödinger equation
    solution = solve_ivp(h, t_span, y0, t_eval=t_eval, method='RK45')
    return np.linalg.norm(solution.y, axis=0) ** 2
    # return solution.y[0]


def main():
    y1 = solve_by_part()
    y2 = directly_solve()

    # Plot the results
    plt.plot(t_eval, y1, label=r'By part')
    plt.plot(t_eval, y2, '--', label=r'Directly')
    # plt.plot(t_eval, np.cos(2*np.pi*t_eval), '*-', label='Exact')

    plt.title(r"RK45 Solution for $y'=i2\pi y$")
    plt.xlabel('t')
    plt.ylabel('$|y|^2$')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()