import numpy as np


s0 = np.eye(2)
sx = np.array([[0, 1], [1, 0]])
sy = np.array([[0, -1j], [1j, 0]])
sz = np.array([[1, 0], [0, -1]])
sp = (sx + 1j * sy) / 2
sm = (sx - 1j * sy) / 2

s = np.array([sx, sy, sz])


def commutator(A, B):
    return A @ B - B @ A


def demo_su2_algebra():
    """
    Demonstrate the su(2) algebra commutation relations.
    """

    def check_zero(mat):
        return np.allclose(mat, np.zeros((2, 2)))

    print("[sx, sy] = 2isz:", check_zero(sx @ sy - sy @ sx - 2j * sz))
    print("[sy, sz] = 2isx:", check_zero(sy @ sz - sz @ sy - 2j * sx))
    print("[sz, sx] = 2isy:", check_zero(sz @ sx - sx @ sz - 2j * sy))

    print("[sp, sm] = sz:", check_zero(sp @ sm - sm @ sp - sz))
    print("[sz, sp] = 2sp:", check_zero(sz @ sp - sp @ sz - 2 * sp))
    print("[sz, sm] = -2sm:", check_zero(sz @ sm - sm @ sz + 2 * sm))


if __name__ == '__main__':
    demo_su2_algebra()
