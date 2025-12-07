import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import ellipeinc
import matplotlib.pyplot as plt

# 参数
a = 1
b = 1.0

# 自变量区间
x = np.linspace(0, 2 * np.pi, 500)

# 被积函数
f = np.sqrt(a ** 2 + b ** 2 * np.cos(x) ** 2)

# 数值积分
I_num = cumulative_trapezoid(f, x, initial=0)

# 椭圆积分形式
A = np.sqrt(a ** 2 + b ** 2)
m = b ** 2 / (a ** 2 + b ** 2)
I_ellip = A * ellipeinc(x, m)  # 不完全椭圆积分第二类

# 比较
plt.figure(figsize=(7, 4))
plt.plot(x, I_num, label='Numerical (trapezoid)', lw=2)
plt.plot(x, I_ellip, '--', label='Analytic (ellipeinc)', lw=2)
plt.xlabel('x')
plt.ylabel('Integral value')
plt.legend()
plt.grid(True)
plt.title('Comparison: Numerical vs Analytic (ellipeinc)')
plt.show()
