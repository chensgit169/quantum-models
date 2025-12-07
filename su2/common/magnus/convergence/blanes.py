from fractions import Fraction
from bernoulli import bernoulli_fraction as bernoulli
from math import factorial, nan


"""
Magnus expansion convergence radius estimation.

Ref:  S. Blanes, F. Casas, J. A. Oteo, and J. Ros. 
Magnus and fer expansions for matrix differential equations: the convergence problem. 
Journal of Physics A: Mathematical and General, 31(1):259, jan 1998

Last modified: 26/10/2025
"""


f_cache = {}
bn_data_filename = "data/b_n.txt"


def f_n_j(n, j):
    """compute f_n^{(j) recursively and store in cache"""
    if (n, j) in f_cache:
        return f_cache[(n, j)]

    if j == 0:
        result = 1 if n == 1 else 0
    else:
        result = 0
        for m in range(1, n - j + 1):
            for p in range(0, m):
                Bp = abs(bernoulli(p))
                term = (Bp / (factorial(p) * m)) * f_n_j(m, p) * f_n_j(n - m, j - 1)
                result += term
                # print(f'{m}-{p}-{term}', end=' ')
        result *= 2

    f_cache[(n, j)] = result
    return result


def b_n(n):
    if n == 0:
        return Fraction(0, 1)

    if n == 1:
        return Fraction(1, 1)

    total = 0
    for p in range(1, n):
        Bp = abs(bernoulli(p))
        total += (Bp / factorial(p)) * f_n_j(n, p)
    _bn = total / n
    return _bn


def ratio_test():
    # estimate convergence radius by b_n / b_{n+1}
    print(f"{'n':>3} {'b_n':>30} {'b_{n+1}':>30} {'b_n / b_{n+1}':>30}")
    for n in range(1, 500):
        bn = b_n(n)
        bn1 = b_n(n + 1)
        ratio = float(bn / bn1) if bn1 != 0 else nan
        print(f"{n:3} ratio={str(ratio)}")


def look_bn():
    bn_list = []
    for n in range(100):
        bn = b_n(n)
        bn_list.append(bn)
        print(f"b_{n} =", bn)

    with open(bn_data_filename, "w") as f:
        for n, bn in enumerate(bn_list):
            f.write(f"b{n}\t{str(bn)}\n")


def load_bn():
    bn_list = []
    with open(bn_data_filename, "r") as f:
        for line in f:
            n_str, bn_str = line.strip().split("\t")
            bn = Fraction(bn_str)
            bn_list.append(bn)
    return bn_list


if __name__ == '__main__':
    look_bn()
    # print(load_bn())
    # print(f_cache)

