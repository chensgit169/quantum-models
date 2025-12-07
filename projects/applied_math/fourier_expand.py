import numpy as np
import matplotlib.pyplot as plt


# 示例函数
def f_test(x):
    return x + x**2  # 定义在 [0, 2pi]


def fourier_series(f, N_samples=1024):
    # 采样点
    x = np.linspace(0, 2 * np.pi, N_samples, endpoint=False)
    y = f(x)
    dx = x[1] - x[0]

    # FFT (近似积分)
    Y = np.fft.fft(y) * dx / np.pi  # 除以 π 来匹配系数定义

    a0 = Y[0].real
    a_n = Y[1:N_samples // 2].real
    b_n = -Y[1:N_samples // 2].imag  # 注意负号

    return a0, a_n, b_n


def demo():
    N_samples = 2048
    a0, a_n, b_n = fourier_series(f_test, N_samples=N_samples)
    x = np.linspace(0, 2 * np.pi, N_samples, endpoint=False)
    y = f_test(x)

    # 用前 M 项重建
    M = 200
    y_approx = a0 / 2 * np.ones_like(x)
    for n in range(1, M + 1):
        y_approx += a_n[n - 1] * np.cos(n * x) + b_n[n - 1] * np.sin(n * x)

    # 绘图
    plt.plot(x, y, label="Original f(x)")
    plt.plot(x, y_approx, label=f"Fourier Series Approx (N={M})")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    demo()
