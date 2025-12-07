import numpy as np
from scipy.integrate import trapezoid


# ======================================================
# Example complex-valued function (replace with your own)
# ======================================================
def v(t):
    # Example: oscillatory complex function
    return np.exp(1j * t ** 2) * np.exp(-0.1 * t)


# ======================================================
# Generate adaptive non-uniform time grid based on |dv/dt|
# ======================================================



# ======================================================
# A1 = ∫ v(t) dt
# ======================================================
def a1(t_i, t_f, n_points=800):
    """Compute A1 = ∫ₜₛₜₐᵣₜᵗₑₙd v(t) dt."""
    t_grid = adaptive_grid(v, t_i, t_f, n_points)
    vals = v(t_grid)
    return trapezoid(vals, t_grid)


# ======================================================
# C2 = ∬ Im[v(t1) v*(t2)] dt2 dt1 (t2 < t1)
# ======================================================
def c2(t_i, t_f, n_points=400):
    """Compute C2 = ∫ dt1 ∫ dt2 Im[v(t1) v*(t2)], with t2 < t1."""
    t_grid = adaptive_grid(v, t_i, t_f, n_points)
    v_vals = v(t_grid)
    inner = np.zeros(n_points, dtype=float)

    for i, t1 in enumerate(t_grid):
        integrand = np.imag(v_vals[i] * np.conj(v_vals[:i + 1]))
        inner[i] = trapezoid(integrand, t_grid[:i + 1])

    return trapezoid(inner, t_grid)


# ======================================================
# A3 = (1/3) ∭ f(t1, t2, t3) dt3 dt2 dt1
# ======================================================



# ======================================================
# Example integrand for A3
# ======================================================
def f_example(t1, t2, t3_array):
    """Example function f(t1, t2, t3)."""
    return np.imag(v(t1) * np.conj(v(t2)) * v(t3_array))


# ======================================================
# Demonstration
# ======================================================
if __name__ == "__main__":
    t_i, t_f = 1.0, 4.0
    print(f"A1({t_i}, {t_f}) =", a1(t_i, t_f))
    print(f"C2({t_i}, {t_f}) =", c2(t_i, t_f))
    print(f"A3({t_i}, {t_f}) =", a3(t_i, t_f, f_example))
