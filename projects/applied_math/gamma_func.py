import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp


mp.dps = 25

g = mp.euler

t_vals = np.linspace(0.001, 100, 2000)

# 计算 Gamma(it) 的相位（辐角）
phase_vals = [mp.arg(mp.gamma(1j * t))- g * t for t in t_vals]

plt.figure(figsize=(8, 5))
plt.plot(t_vals, phase_vals)
plt.xlabel("t (for z = i t)")
plt.ylabel("arg Γ(i t)")
plt.title("Phase of Gamma(i t) along the imaginary axis")
plt.grid(True)
plt.show()
