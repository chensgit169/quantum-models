import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root

# =====================================================
# Parameters
# =====================================================
E = 1.0
p = -4  # <-- new parameter here

class Stokes:

    def __init__(self, w_func):
        self.w = w_func

    def f_turning(self, z):
        z = z[0] + 1j * z[1]
        val = self.w(z)**2
        return [val.real, val.imag]

    def search_turning_points(self, guesses):
        turning_points = []

        for g in guesses:
            sol = root(self.f_turning, g, tol=1e-12)
            if sol.success:
                t0 = sol.x[0] + 1j * sol.x[1]
                # avoid duplicates
                if all(abs(t0 - tp) > 1e-6 for tp in turning_points):
                    turning_points.append(t0)
        return turning_points

    # =====================================================
    # Anti-Stokes ODE: dt/ds = conjugate(w)/|w|
    # =====================================================
    def ode(self, s, y):
        t = y[0] + 1j * y[1]
        wt = self.w(t)
        v = np.conj(wt) / np.abs(wt)
        return [v.real, v.imag]

    def draw_anti_stokes_lines(self, t0, eps = 1e-4):

        s_max = 1

        plt.figure(figsize=(8, 10))

        angles = initial_angles_at(t0)
        for theta in angles:
            t_init = t0 + eps * np.exp(1j * theta)
            y0 = [t_init.real, t_init.imag]

            sol = solve_ivp(
                self.ode,
                [0, s_max],
                y0,
                max_step=0.1,
                rtol=1e-9,
                atol=1e-9
            )


def w_test(t):
    t = t.astype(complex)
    return np.sqrt(1 + (p + E * np.tanh(t)) ** 2)


# =====================================================
# Initial ray angles from WKB local formula:
#   θ = 2/3 (mπ - argβ),   m=0,1,2
#
# We approximate β = sqrt(f'(t0))
# =====================================================
def fprime(t):
    return 2 * (p + E * np.tanh(t)) * E * (1 / np.cosh(t)) ** 2


def initial_angles_at(t0):
    beta = np.sqrt(fprime(t0))
    argb = np.angle(beta)
    angs = []
    for m in [0, 1, 2]:
        theta = (2 / 3) * (m * np.pi - argb)
        angs.append(theta)
    return angs



# =====================================================
# Integrate & plot Anti-Stokes lines
# =====================================================

# def demo():
    #         T = sol.y[0] + 1j * sol.y[1]
    #         plt.plot(T.real, T.imag, '-', lw=1.3)
    #
    # # =====================================================
    # # Mark turning points
    # # =====================================================
    # for tp in turning_points:
    #     plt.plot(tp.real, tp.imag, 'bo', markersize=10, label="turning point")
    #
    # # Poles of tanh: iπ/2 + iπk
    # for k in range(-1, 1):
    #     pole = 1j * (np.pi / 2 + np.pi * k)
    #     plt.plot(pole.real, pole.imag, 'rx', markersize=10)
    #
    # plt.xlabel("Re(t)")
    # plt.ylabel("Im(t)")
    # plt.title(f"Anti-Stokes lines for w(t)=sqrt(1 + (p+E*tanh t)^2),  E={E}, p={p}")
    # plt.grid(True)
    # plt.axhline(0, color='k', linewidth=0.5)
    # plt.axvline(0, color='k', linewidth=0.5)
    # plt.show()


if __name__ == '__main__':
    draw_anti_stokes_lines()