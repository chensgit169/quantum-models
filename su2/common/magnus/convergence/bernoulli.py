import sympy
from fractions import Fraction
from math import factorial


"""
Bernoulli number generator.

B0 = 1, B1 = -1/2, B2 = 1/6, B4 = -1/30, B6 = 1/42, B8 = -1/30, B10 = 5/66, ...
B(odd n > 1) = 0

Last modified: 22/08/2025
"""


def bernoulli_fraction(n):
    """obtain Bernoulli numbers as Fraction objects"""
    b = sympy.bernoulli(n)

    # note that in sympy b1=1/2 (first Bernoulli numbers)
    if n == 1:
        b = -b

    bf = Fraction(int(b.p), int(b.q))  # b.p numerator, b.q dominator
    return bf


def expansion():
    """
    verify x/(exp(x)-1) = sum_n=0 B_n * x**n / n!
    """
    n_lim = 20

    x = sympy.symbols('x')
    f = x / (sympy.exp(x)-1)
    series = f.series(x, 0, n_lim)
    for n in range(n_lim):
        assert series.coeff(x, n) == bernoulli_fraction(n)/factorial(n)


def demo():
    for n in range(11):
        print(f"B_{n} =", bernoulli_fraction(n))


if __name__ == '__main__':
    demo()
