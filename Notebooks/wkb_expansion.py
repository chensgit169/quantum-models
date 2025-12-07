from sympy import symbols, Function, sqrt, diff, simplify, expand, pprint


def compute_p_list(Qexpr, x, N, branch=1):
    """
    Compute [P_-1, P_0, ..., P_N] based on the recursive relations.

    Parameters
    ----------
    Qexpr  : SymPy expression, Q(x)
    x      : Sympy variable
    N      : maximum index m (computes up to P_N)
    branch : sign choice for P_-1 = branch*sqrt(Q), branch = +1 or -1
    """
    # Initial condition: P_-1 = ±sqrt(Q)
    P = {-1: branch * sqrt(Qexpr)}

    # Order O(hbar^1): 2 P_-1 P_0 + dP_-1/dx = 0 → P_0 = -P'_-1/(2 P_-1)
    P[0] = -diff(P[-1], x) / (2 * P[-1])

    # Recursive relation for m ≥ 0:
    # 2 P_-1 P_{m+1} + (Σ_{l=0}^m P_l P_{m-l}) * dP_m/dx = 0
    # → P_{m+1} = -(Σ P_l P_{m-l}) * P'_m / (2 P_-1)
    for m in range(0, N):
        Sm = sum(P[l] * P[m - l] for l in range(0, m + 1))
        P[m + 1] = simplify(-(Sm * diff(P[m], x)) / (2 * P[-1]))

    # Return as a list ordered from P_-1 to P_N
    return [P[k] for k in range(-1, N + 1)]


# # Example 1: Keep Q(x) general, compute up to P_3
# P_list_general = compute_p_list(Q, 3, branch=1)
# print("General Q(x):")
# for i, p in enumerate(P_list_general, start=-1):
#     print(f"P_{i}(x) =")
#     pprint(p, use_unicode=True)
#     print()
#
# # Example 2: Use specific Q(x) = x^2 + 1, compute up to P_3
# Q_example = x ** 2 + 1
# P_list_example = compute_p_list(Q_example, 3, branch=1)
# print("Example Q(x) = x^2 + 1:")
# for i, p in enumerate(P_list_example, start=-1):
#     print(f"P_{i}(x) =")
#     pprint(simplify(expand(p)))
#     print()
