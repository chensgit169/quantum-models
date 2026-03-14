import numpy as np
from scipy import special

# def integral_numeric(g):
#     # 被积函数
#     f = lambda t: np.exp(1j * g * np.sin(t)) / 2
#     # 分别对实部和虚部积分（quad 不直接支持复数）
#     real = integrate.quad(lambda t: np.real(f(t)), 0, np.pi)[0]
#     imag = integrate.quad(lambda t: np.imag(f(t)), 0, np.pi)[0]
#     return real + 1j * imag
#
# # 测试几个 g 值
# g_values = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
#
# print(f"{'g':>6} | {'numeric':>20} | {'exact':>20} | {'error':>10}")
# print("-" * 70)
#
# for g in g_values:
#     num = integral_numeric(g)
#     exact = 0.5 * np.pi * (special.j0(g) + 1j * special.struve(0, g))
#     error = abs(num - exact)
#     print(f"{g:6.2f} | {num:20.12e} | {exact:20.12e} | {error:10.2e}")


def plot():
    import matplotlib.pyplot as plt

    g_values = np.linspace(0, 40, 200)
    j0_vals = special.j0(g_values)
    h0_vals = special.struve(0, g_values)
    abs_vals = np.pi / 2 * np.abs(j0_vals + 1j * h0_vals)

    plt.figure(figsize=(8, 5))

    # plt.plot(g_values, j0_vals/2, label=r'$J_0(g)$')
    # r = np.pi / 2 * j0_vals * np.sin(abs_vals)/abs_vals
    # plt.plot(g_values, np.arcsin(r) / np.pi, label='eps_1st')

    plt.plot(g_values, j0_vals, label=r'$\Delta J_0(g)$')
    plt.plot(g_values, np.arcsin(np.pi / 2 * j0_vals)/np.pi * 2)

    # plt.plot(g_values, h0_vals, label=r'$H_0(g)$')
    # plt.plot(g_values, np.abs(j0_vals + 1j * h0_vals), label=r'abs')

    # plt.plot(g_values, np.sin(abs_vals)/abs_vals)


    plt.xlabel(r'$g$')
    plt.ylabel('Function Values')
    plt.title('Bessel J0 and Struve H0 Functions')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot()
