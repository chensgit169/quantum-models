import numpy as np
from scipy.linalg import expm

import matplotlib.pyplot as plt


# Hamiltonian
d = 1
o = 2 * d
H = np.array([[d, o], [o, -d]])

# Rabi frequency
w = np.sqrt(o ** 2 + d ** 2)


def demo():
    # initial state
    psi_0 = np.array([1, 0])

    # time evolution
    ts = np.linspace(0, 4 * np.pi, 400)
    psi = np.array([expm(-1j * H * t) @ psi_0 for t in ts])

    a = psi[:, 0]

    plt.plot(ts, np.real(a), label=r'$\Re(\alpha)$')
    plt.plot(ts, np.imag(a), label=r'$\Im(\alpha)$')
    plt.plot(ts, np.abs(a) ** 2, label=r'$|\alpha|^2$')
    plt.legend()
    plt.show()


def verify():
    # initial state
    psi_0 = np.array([1, 0])

    # time evolution (by exponent computation)
    ts = np.linspace(0, 4 * np.pi, 400)
    # psi = np.array([expm(-1j * H * t) @ psi_0 for t in ts])
    psi = compute(psi_0, ts)

    a = psi[:, 0]

    # analytical solution for probability to stay
    cos2t = ((d + w) ** 2 - o ** 2) / ((d + w) ** 2 + o ** 2)
    p = np.cos(w * ts) ** 2 + cos2t ** 2 * np.sin(w * ts) ** 2

    plt.plot(ts, p, label=r'$P(t)$')
    plt.plot(ts, np.abs(a) ** 2, label=r'$|\alpha|^2$')

    print('mean error: ', np.mean(np.abs(np.abs(a) ** 2-p)))

    plt.legend()
    plt.show()


def compute(psi_0, ts):
    # compute for arbitrary initial state
    cost = (d + w)/np.sqrt((d+w)**2+o**2)
    sint = o / np.sqrt((d + w) ** 2 + o ** 2)

    # eigen states
    psi_p = np.array([cost, sint])
    psi_m = np.array([-sint, cost])

    cp = psi_p.conjugate().T @ psi_0
    cm = psi_m.conjugate().T @ psi_0

    psi = np.exp(-1j*w*ts).reshape(-1, 1) * cp * psi_p + np.exp(+1j*w*ts).reshape(-1, 1) * cm * psi_m
    return psi


if __name__ == '__main__':
    verify()
