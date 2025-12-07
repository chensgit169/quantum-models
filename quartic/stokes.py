import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root

# =====================================================
# Parameters
# =====================================================
E = 1.0
p = 4  # <-- new parameter here


def w(t):
    t = t.astype(complex)
    return np.sqrt(1 + (p + E * np.tanh(t)) ** 2)


def get_turning_points():
    def arctanh(z):
        return 0.5 * np.log((1 + z) / (1 - z))

    z_plus = (-p + 1j) / E
    z_minus = (-p - 1j) / E

    t_plus = arctanh(z_plus)
    t_minus = arctanh(z_minus)

    turning_points = [t_plus, t_minus]
    return turning_points


# =====================================================
# Initial ray angles from WKB local formula:
#   θ = 2/3 (mπ - argβ),   m=0,1,2
#
# We approximate β = sqrt(f'(t0))
# =====================================================
def fprime(t):
    return 2 * (p + E * np.tanh(t)) * E * (1 / np.cosh(t)) ** 2


# ----------------------------
# Airy-based initial angles
# ----------------------------
def initial_angles_at(t0):
    """
    Compute Anti-Stokes angles in t-plane using local Airy expansion
    """
    fp = fprime(t0)
    arg_fp = np.angle(fp)

    # Standard Airy Anti-Stokes angles: 0, 2pi/3, -2pi/3
    theta_airy_as = [0, 2 * np.pi / 3, -2 * np.pi / 3]

    # Convert to t-plane
    theta_t_as = [th - arg_fp / 3 for th in theta_airy_as]
    return theta_t_as


# ----------------------------
# ODE with local Airy direction
# ----------------------------
def ode_with_local(s, y, sign):
    t = y[0] + 1j * y[1]
    wt = w(t)

    v = sign * np.conj(wt) / abs(wt)

    return [v.real, v.imag]


# ----------------------------
# Draw Anti-Stokes lines using Airy directions
# ----------------------------
def draw_anti_stokes_lines(t0):
    eps = 1e-5
    s_max = 0.4

    angles = initial_angles_at(t0)

    for theta in angles:
        print("theta/pi =", theta / np.pi)

        t_init = t0 + eps * np.exp(1j * theta)
        y0 = [t_init.real, t_init.imag]
        # plt.plot(t_init.real, t_init.imag, 'go')  # initial point

        # decide initial direction
        v_local = np.exp(1j * theta)
        wt = w(t_init)
        ratio = v_local / (np.conj(wt) / abs(wt))
        print(ratio)  # very close to 1 or -1
        direction = np.sign(ratio.real)

        sol = solve_ivp(
            lambda s, y: ode_with_local(s, y, direction),
            [0, s_max],
            y0,
            max_step=0.001,
            rtol=1e-9, atol=1e-10
        )

        T = sol.y[0] + 1j * sol.y[1]

        # Plot the line
        plt.plot(T.real, T.imag, '-', lw=1.3)


def demo():
    turning_points = get_turning_points()[:1]

    plt.figure(figsize=(8, 10))

    for tp in turning_points:
        draw_anti_stokes_lines(tp)
    # =====================================================
    # Mark turning points
    # =====================================================
    for tp in turning_points:
        plt.plot(tp.real, tp.imag, 'bo', markersize=10, label="turning point")

    # Poles of tanh: iπ/2 + iπk
    for k in range(0, 1):
        pole = 1j * (np.pi / 2 + np.pi * k)
        plt.plot(pole.real, pole.imag, 'rx', markersize=10)

    plt.xlabel("Re(t)")
    plt.ylabel("Im(t)")
    plt.title(f"Anti-Stokes lines for w(t)=sqrt(1 + (p+E*tanh t)^2),  E={E}, p={p}")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    demo()
