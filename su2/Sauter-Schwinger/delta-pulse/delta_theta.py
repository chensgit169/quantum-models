import numpy as np
import matplotlib.pyplot as plt


from i_integrals import delta_theta

# 参数
d = 1.0
x = np.linspace(-15, 15, 1000)

# 原函数
y = delta_theta(x, d)

# 上界（分段定义更直观）
upper = np.zeros_like(x)
upper[x < 0] = d / (1 + x[x < 0]**2)
upper[(x >= 0) & (x <= d)] = min(d, np.pi)
upper[x > d] = d / (1 + (x[x > d] - d)**2)

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(x, y, label=r'$f(x)=\arctan(x)-\arctan(x-d)$', linewidth=2)
plt.plot(x, upper, 'r--', label='Upper bound', linewidth=2)
plt.vlines(d/2, 0, 1.1*d, colors='gray', linestyles='dotted')

plt.ylim(0, d*1.1)
plt.title(r'$\arctan(x)-\arctan(x-d)$ and its upper bound')
plt.xlabel('x')
plt.ylabel('value')
plt.grid(True)
plt.legend()
plt.show()
