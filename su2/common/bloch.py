import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

"""
Demonstrate the evolution of a qubit on the Bloch sphere under a constant magnetic field.

Last modified: 2025-11-30
"""


def bloch_vector(alpha, beta):
    """Compute the Bloch sphere vector from spinor components a and b."""
    norm = np.sqrt(np.abs(alpha) ** 2 + np.abs(beta) ** 2)
    alpha, beta = alpha / norm, beta / norm
    x = 2 * np.real(np.conj(alpha) * beta)
    y = 2 * np.imag(np.conj(alpha) * beta)
    z = np.abs(alpha) ** 2 - np.abs(beta) ** 2
    return np.array([x, y, z])


def demo():
    # ---------------------------
    # 时间演化参数
    # ---------------------------
    T = 10.0
    N = 500
    t = np.linspace(0, T, N)

    # 恒定磁场 (Bx, By, Bz)
    B = np.array([1.0, 0.5, 1.0])

    def H(a, b):
        """哈密顿量作用"""
        return np.array([B[2] / 2 * a + (B[0] - 1j * B[1]) / 2 * b,
                         (B[0] + 1j * B[1]) / 2 * a - B[2] / 2 * b])

    # 初态 |up>
    psi = np.zeros((N, 2), dtype=complex)
    psi[0] = np.array([1.0, 0.0])

    # Euler 法演化
    dt = t[1] - t[0]
    for i in range(N - 1):
        psi[i + 1] = psi[i] - 1j * dt * H(psi[i, 0], psi[i, 1])

    # Bloch 球向量

    return psi


def make_movie(psi, filename=None, N=None, theta_lim=np.pi):
    if N is None:
        N = psi.shape[0]

    bloch_vecs = np.array([bloch_vector(a, b) for a, b in psi])

    phases = np.angle(psi[:, 0] / (psi[:, 1] + 1e-10))

    # ---------------------------
    # 绘制 Bloch 球（限制到 theta_lim）
    # ---------------------------
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])

    # 绘制限制后的半透明球面
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, theta_lim, 50)  # ★ 这里限制 θ 角度范围
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(
        x_sphere, y_sphere, z_sphere,
        color='c', alpha=0.1, edgecolor='k'
    )

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([np.cos(theta_lim), 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_box_aspect([1, 1, (1 - np.cos(theta_lim)) / 2])  # 让 xyz 比例一致

    point, = ax.plot([], [], [], 'o', markersize=8, color='r')
    line, = ax.plot([], [], [], '-', lw=2, color='r')

    # 用颜色表示相位
    norm_phase = (phases + np.pi) / (2 * np.pi)  # 标准化到 [0,1]

    def update(frame):
        # 当前点
        point.set_data([bloch_vecs[frame, 0]], [bloch_vecs[frame, 1]])
        point.set_3d_properties([bloch_vecs[frame, 2]])
        # 当前轨迹
        line.set_data(bloch_vecs[:frame + 1, 0], bloch_vecs[:frame + 1, 1])
        line.set_3d_properties(bloch_vecs[:frame + 1, 2])
        # 根据相位改变点的颜色
        point.set_color(plt.cm.hsv(norm_phase[frame]))

        return point, line

    ani = FuncAnimation(fig, update, frames=N, interval=50, blit=True)

    if filename:
        ani.save(filename, writer=PillowWriter(fps=20))


def xy_trajectory(psi):
    bloch_vecs = np.array([bloch_vector(a, b) for a, b in psi])
    xs, ys = bloch_vecs[:, 0], bloch_vecs[:, 1]
    return xs, ys


def plot_xy_trajectory(psi):
    bloch_vecs = np.array([bloch_vector(a, b) for a, b in psi])
    xs, ys = bloch_vecs[:, 0], bloch_vecs[:, 1]
    plt.figure(figsize=(6, 6))
    plt.plot(xs, ys)
    plt.plot(0, 0, 'rx')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Bloch Sphere XY Projection')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    return xs, ys


if __name__ == '__main__':
    psi = demo()
    # make_movie(psi, filename='bloch_sphere_evolution.gif')
    plot_xy_trajectory(psi)
