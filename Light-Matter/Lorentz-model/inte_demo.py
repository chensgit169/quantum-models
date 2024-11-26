import numpy as np
from scipy.integrate import cumulative_trapezoid as cumtrapz


# 定义 F(x) 函数
def F(x):
    return np.sin(x)  # 示例函数，可以替换成任意需要的 F(x)


# 创建自变量数组 xs
xs = np.linspace(0, 10, 100)  # 从 0 到 10，生成 100 个点

# 计算积分的累积值，cumtrapz 返回的是积分值数组的 n-1 项，需要在头部插入一个 0
ys = cumtrapz(F(xs), xs, initial=0)

print("xs:", xs)
print("ys:", ys)
