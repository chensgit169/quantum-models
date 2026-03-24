import numpy as np
import matplotlib.pyplot as plt

# 定义 z 的取值范围
z = np.linspace(0, 4, 400)

# 计算 arcsin(z)
y = np.arcsin(z.astype(np.complex128))  # NumPy 支持复数运算

# 提取实部和虚部
real_y = np.real(y)
imag_y = np.imag(y)

# 绘制实部和虚部
plt.figure(figsize=(8, 5))
plt.plot(z, real_y/np.pi, label=r'Re(arcsin(z))')
plt.plot(z, imag_y/np.pi, label=r'Im(arcsin(z))')

# draw vertical line at y=1
plt.axvline(1, color='gray', linestyle='--', label='y=1')

plt.title('complex-valued f(z)=arcsin(z)')
plt.xlabel('z')
plt.ylabel(r'f(z)/$\pi$')
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig('figures/arcsin_complex.png', dpi=400)