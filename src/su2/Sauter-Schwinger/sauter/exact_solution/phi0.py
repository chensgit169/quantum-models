import numpy as np
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

# ---------------------
# 参数设置
# ---------------------
E0 = 2.0
eps = 2.22e-16  # 机器精度

# p 的范围
p_values = np.linspace(-50, 50, 101)

# 截断上限 T （按最大 |p| 估算安全）
p_max = np.max(np.abs(p_values))
T = 0.5 * np.log(E0 * (p_max + E0) / eps)
T += 2  # 安全裕度
print("Using truncation T =", T)

# 网格
N = 200000
xs = np.linspace(0, T, N)


# ---------------------
# 积分函数
# ---------------------
def f(x, p):
    return np.sqrt(1 + (p + E0) ** 2) - np.sqrt(1 + (p + E0 * np.tanh(x)) ** 2)


# ---------------------
# 对每个 p 做积分
# ---------------------
I_values = []
for p in p_values:
    vals = f(xs, p)
    I = trapezoid(vals, xs)
    I_values.append(I)
I_values = np.array(I_values)

# ---------------------
# 绘图
# ---------------------
plt.figure(figsize=(8, 4))
plt.plot(p_values, I_values, lw=2)
plt.xlabel("p")
plt.ylabel("Integral I(p)")
plt.title("Integral of sqrt(1+(p+E0)^2) - sqrt(1+(p+E0 tanh(x))^2) vs p")
plt.grid(True)
plt.show()
