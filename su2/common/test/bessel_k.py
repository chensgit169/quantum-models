import numpy as np
import matplotlib.pyplot as plt
from scipy.special import k0, k1

# 定义 x 轴
x = np.linspace(1e-3, 10, 400)

# Euler 常数
gamma = 0.5772156649

# ==== K0 ====
K0 = k0(x)
# 渐近式
K0_small = -np.log(x/2) - gamma
K0_large = np.sqrt(np.pi/(2*x)) * np.exp(-x)

plt.figure(figsize=(7,5))
plt.plot(x, K0, label=r"$K_0(x)$")
plt.plot(x, K0_small, '--', label=r"$-\ln(x/2)-\gamma$")
plt.plot(x, K0_large, '--', label=r"$\sqrt{\frac{\pi}{2x}}\,e^{-x}$")
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$x$")
plt.ylabel(r"$K_0(x)$")
# plt.title(r"Modified Bessel function $K_0(x)$ and asymptotic expressions")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.savefig('figures/bessel_K0.pdf', dpi=400)
plt.show()

# ==== K1 ====
K1 = k1(x)
# 渐近式
K1_small = 1/x
K1_large = np.sqrt(np.pi/(2*x)) * np.exp(-x) # * (1 + 3/(8*x))

plt.figure(figsize=(7,5))
plt.plot(x, K1, label=r"$K_1(x)$")
plt.plot(x, K1_small, '--', label=r"$1/x$")
# plt.plot(x, K1_large, '--', label=r"$\sqrt{\frac{\pi}{2x}}\,e^{-x}\!\left(1+\frac{3}{8x}\right)$")
plt.plot(x, K1_large, '--', label=r"$\sqrt{\frac{\pi}{2x}}\,e^{-x}$")
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$x$")
plt.ylabel(r"$K_1(x)$")
# plt.title(r"Modified Bessel function $K_1(x)$ and asymptotic expressions")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.savefig('figures/bessel_K1.pdf', dpi=400)
plt.show()
