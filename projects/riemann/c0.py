import numpy as np
import matplotlib.pyplot as plt


# Define the function f(x) = cos(2π(x^2 - x - 1/16)) / cos(2πx)
def f(x):
    """Safely compute f(x). Avoid division by very small values of cos(2πx)."""
    num = np.cos(2 * np.pi * (x ** 2 - x - 1 / 16.0))
    den = np.cos(2 * np.pi * x)
    return num / den


# Singular points where cos(2πx) = 0 in [0,1]: x = 1/4 and 3/4
singular_points = np.array([0.25, 0.75])

# Create a grid of points, avoiding values too close to singular points
N = 2000
x = np.linspace(0, 1, N)
tol = 1e-6
mask_safe = np.ones_like(x, dtype=bool)
for sp in singular_points:
    mask_safe &= np.abs(x - sp) > tol

x_safe = x[mask_safe]
y_safe = f(x_safe)


# Estimate the limit at singular points using symmetric points around x0
def estimate_limit_at(x0, h_list=(1e-3, 1e-4, 1e-5, 1e-6)):
    vals = []
    for h in h_list:
        xl = x0 - h
        xr = x0 + h
        if xl < 0 or xr > 1:
            vals.append(np.nan)
            continue
        vl = f(np.array([xl]))[0]
        vr = f(np.array([xr]))[0]
        vals.append(0.5 * (vl + vr))
    vals = np.array(vals)
    if np.all(np.isfinite(vals)):
        if np.abs(vals[-1] - vals[-2]) < 1e-3:
            return float(vals[-1])
        else:
            return float(vals[-1])
    else:
        return np.nan


# Compute estimated limits at singular points
limits = {}
for sp in singular_points:
    limits[sp] = estimate_limit_at(sp)

# Plot the function
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(x_safe, y_safe, label='f(x) on [0,1] excluding singular points')

# Mark the singular points and their estimated limits
for sp in singular_points:
    ax.axvline(sp, linestyle='dashed', linewidth=1)
    lim = limits[sp]
    if np.isfinite(lim):
        ax.plot(sp, lim, marker='o', markersize=6, label=f'limit at x={sp} ≈ {lim:.6g}')
    else:
        ax.plot(sp, 0, marker='x', markersize=6, label=f'no stable limit at x={sp}')

ax.set_xlim(0, 1)
ax.set_xlabel('x')
ax.set_ylabel('f(x) = cos(2π(x² - x - 1/16)) / cos(2πx)')
ax.set_title('Plot of f(x) on [0,1] with singular handling')
ax.legend(loc='upper right', fontsize='small')
ax.grid(True)

plt.show()

# Print estimated limits for reference
print("Estimated limits at singular points (x where cos(2πx)=0):")
for sp, val in limits.items():
    print(f"  x = {sp}: estimated limit ≈ {val}")
