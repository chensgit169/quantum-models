import numpy as np
import matplotlib.pyplot as plt
from scipy.special import i0
from scipy.integrate import quad


func_name = r'Wigner Sunrise $f(s) \propto s I_0\left(\frac{s^2}{8}\right) e^{-\frac{3s^2}{8}}$'


def wigner_sunrise(s):
    return s * i0(s**2 / 8) * np.exp(-s ** 2 * 3 / 8)


# Calculate the normalization constant by numerical integration
res = quad(wigner_sunrise, 0, 20)
norm, error = res

s_values = np.linspace(0, 5, 500)  # 取 s > 0
f_values = wigner_sunrise(s_values) / norm


# plot
plt.figure(figsize=(8, 6))
plt.plot(s_values, f_values)
plt.xlim(left=0)
plt.ylim(bottom=0)

plt.xlabel(r'$s$', fontsize=14)
plt.ylabel(r'$f(s)$', fontsize=14)
plt.title(func_name, fontsize=16)
# plt.legend()
plt.grid(True)
plt.savefig("analytic.png", dpi=400)
