import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import bisect


def find_roots(xs, func):
    fs = np.array([func(x) for x in xs])

    sign = fs > 0
    cross = ~(sign[:-1] == sign[1:])

    a_set = xs[:-1][cross]
    b_set = xs[1:][cross]
    roots = []

    for a, b in zip(a_set, b_set):
        root = bisect(func, a, b)
        roots.append(root)

    return np.array(roots), fs


def demo():
    def f(x):
        return np.sin(x) + 0.5
    xs, fs = np.linspace(0, 20, 100)
    roots = find_roots(xs, f)
    plt.plot(xs, fs)
    plt.plot(roots, np.zeros_like(roots), '*')
    plt.show()
