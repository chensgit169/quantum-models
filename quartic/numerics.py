from math import factorial, pi

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.linalg import eigh
from scipy.special import hermite


L = 8  # 截断区间 [-L, L]
N = 2000  # 网格点数
x = np.linspace(-L, L, N)
dx = x[1] - x[0]
num_levels = 3  # 求前 10 个能级

# =======================
# 方法1：有限差分法
# =======================
# 构造差分矩阵
diagonal = 2.0 / dx ** 2 + x ** 4
off_diagonal = -1.0 / dx ** 2 * np.ones(N - 1)
H_fd = np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)

# 求本征值和本征向量
E_fd, psi_fd = eigh(H_fd)

print("=== 有限差分法前10个本征能级 ===")
for i in range(num_levels):
    print(f"n={i}, E={E_fd[i]:.6f}")

# 归一化波函数
psi_fd_norm = psi_fd / np.sqrt(dx)


# =======================
# 方法2：谱方法（谐振子基底展开）
# =======================
def harmonic_basis(n, x):
    """一维谐振子本征函数，hbar=1, m=1/2, ω=1"""
    Hn = hermite(n)
    psi_n = Hn(np.sqrt(1) * x) * np.exp(-0.5 * x ** 2)
    norm = 1.0 / np.sqrt(2 ** n * factorial(n) * np.sqrt(pi))
    return norm * psi_n


# 构造谱方法矩阵（有限个基底）
N_basis = 80
H_spec = np.zeros((N_basis, N_basis))
for m in range(N_basis):
    for n in range(N_basis):
        # 矩阵元 <m|x^4|n>
        # 积分用数值积分
        psi_m = harmonic_basis(m, x)
        psi_n = harmonic_basis(n, x)
        H_spec[m, n] = trapezoid(psi_m * x ** 4 * psi_n, x)
        if m == n:
            H_spec[m, n] += (2 * n + 1)  # 谐振子能量 E_n = 2n+1 (单位化)

# 求本征值
E_spec, psi_spec_coeff = eigh(H_spec)
print("\n=== 谱方法 ===")
for i in range(num_levels):
    print(f"n={i}, E={E_spec[i]:.6f}")

# 构造波函数
psi_spec = np.zeros((num_levels, N))
for i in range(num_levels):
    for n in range(N_basis):
        psi_spec[i, :] += psi_spec_coeff[n, i] * harmonic_basis(n, x)

# =======================
# 绘图：波函数比较
# =======================
plt.figure(figsize=(10, 6))
for i in range(num_levels):
    plt.plot(x, psi_fd_norm[:, i], label=f'FD n={i}')
plt.title("Quartic Oscillator Wavefunctions (Finite Difference)")
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
for i in range(num_levels):
    plt.plot(x, psi_spec[i, :], label=f'Spectral n={i}')
plt.title("Quartic Oscillator Wavefunctions (Spectral Method)")
plt.xlabel("x")
plt.ylabel("ψ(x)")
plt.legend()
plt.show()
