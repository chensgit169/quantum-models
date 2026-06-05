import numpy as np
import matplotlib.pyplot as plt


# parameters
g = 1.5         # coupling strength
omega = 1.0      # driving frequency


def eps():
    # d range
    d = np.linspace(0, 10, 500)

    # quasienergies
    epsilon = 0.5 * np.sqrt((d - omega)**2 + g**2)

    # plot
    plt.figure()
    plt.plot(d,  epsilon)
    plt.plot(d, -epsilon)
    plt.xlabel(r"$d$")
    plt.ylabel(r"Quasienergy $\varepsilon$")
    plt.title("Floquet quasienergy spectrum")
    plt.show()


def eps_int():
    # d range
    d = np.linspace(0, 12, 500)

    # quasienergies
    omega = np.sqrt(d**2+g**2)
    cos = np.cos(d * np.pi) * np.cos(omega * np.pi) - (d / omega) * np.sin(d * np.pi) * np.sin(omega * np.pi)
    epsilon = np.arccos(cos) / (2 * np.pi)

    # plot
    plt.figure()
    plt.plot(d, epsilon)
    plt.plot(d, -epsilon)
    plt.hlines(y=0.5, xmin=np.min(d), xmax=np.max(d), colors='gray', linestyles='dashed', alpha=0.5)
    plt.hlines(y=-0.5, xmin=np.min(d), xmax=np.max(d), colors='gray', linestyles='dashed', alpha=0.5)
    plt.xlabel(r"$d$")
    plt.ylabel(r"Quasienergy $\varepsilon$")
    plt.title("Floquet quasienergy spectrum")
    plt.show()


if __name__ == '__main__':
    # eps()
    eps_int()