import numpy as np

# 已知的数据点
xp = np.array([0, 1, 2, 3])   # x 坐标
fp = np.array([0, 2, 4, 6])   # y 坐标

# 需要插值的点
x = np.array([0.5, 1.5, 2.5])

# 使用 np.interp 进行插值
y = np.interp(x, xp, fp)

print("插值结果:", y)
