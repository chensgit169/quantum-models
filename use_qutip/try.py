import numpy as np
import matplotlib.pyplot as plt
from qutip import sigmax, basis, sigmay, sigmaz, mesolve

# 定义二能级系统的哈密顿量
omega = 1.0  # 能级之间的能隙
H = omega * sigmax() / 2.0

# 初始态为|0⟩
psi0 = basis(2, 0)

# 时间演化(Master equation evolution)
tlist = np.linspace(0, 10, 100)
result = mesolve(H, psi0, tlist, [], [sigmax(), sigmay(), sigmaz()])

# 绘制期望值随时间的演化
plt.plot(tlist, result.expect[0], label="X")
plt.plot(tlist, result.expect[1], label="Y")
plt.plot(tlist, result.expect[2], label="Z")
plt.xlabel('Time')
plt.ylabel('Expectation values')
plt.legend()
plt.show()
