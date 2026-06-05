import os

import numpy as np
from tqdm import tqdm

from exact_solution import quasi_energy
import matplotlib.pyplot as plt

plt.rcParams['font.size'] = 18


def shirley_1965_fig1():
    import matplotlib.pyplot as plt

    data_filename = 'data/shirley_1965_fig1.npz'
    if os.path.exists(data_filename):
        data = np.load(data_filename)
        v_vals = data['v']
        e_vals = data['e']
        f = data['f']
    else:
        f = 1
        v_vals = np.linspace(0, 7.5, 2001)
        e_vals = np.array([quasi_energy(f, v) for v in tqdm(v_vals)])
        np.savez(data_filename, v=v_vals, e=e_vals, f=f)

    line1 = plt.plot(v_vals, - e_vals, '--', label=r'$\epsilon_-$')
    color1 = line1[0].get_color()
    line2 = plt.plot(v_vals, e_vals, label=r'$\epsilon_+$')
    color2 = line2[0].get_color()

    plt.plot(v_vals, e_vals + 1, color=color2)
    plt.plot(v_vals, - e_vals + 1, '--', color=color1)
    plt.plot(v_vals, e_vals + 2, color=color2)
    plt.plot(v_vals, - e_vals + 2, '--', color=color1)

    circle_x, circle_y = 2.904, 0.5
    circle = plt.Circle((circle_x, circle_y), radius=0.1, linestyle=':',
                        color='black', fill=False, linewidth=1)
    plt.gca().add_patch(circle)
    plt.text(circle_x + 0.2, circle_y, 'avoided\ncrossing', color='black', fontsize=17,
             verticalalignment='center')

    circle_x, circle_y = 3.95, 0
    circle = plt.Circle((circle_x, circle_y), radius=0.1, linestyle=':',
                        color='black', fill=False, linewidth=1)
    plt.gca().add_patch(circle)
    plt.text(circle_x + 0.2, circle_y, 'exact\ncrossing', color='black', fontsize=17,
             verticalalignment='center')

    plt.xlabel(r'$\Delta$')
    plt.ylabel(r'$\epsilon$')
    # plt.legend()
    plt.tight_layout()
    plt.savefig('figures/Shirley/shirley_1965_fig1.pdf', dpi=400)


def shirley_1965_fig2():
    import matplotlib.pyplot as plt

    v_points = []
    v_points += [0]
    v_points += [2.9045, 2.906]
    v_points += [4.9476, 4.948]
    v_points += [6.963, 6.964, 7.5]

    data_filename = 'data/shirley_1965_fig2.npz'

    f = 1
    dv = 1e-5

    def e_v(v):
        return quasi_energy(f, v)

    def de_v(v):
        return (e_v(v + dv) - e_v(v - dv)) / (2 * dv)

    # v_vals = np.linspace(0, 7, 2001)
    # roots, e_vals = find_roots(v_vals, de_v)
    # for r in roots:
    #     print(f'root at v={r:.7f}, e={de_v(r):.7f}')

    if os.path.exists(data_filename):
        data = np.load(data_filename)
        v_vals = data['v']
        de_vals = data['de']
    else:
        v_vals = []
        de_vals = []

        for i, v in enumerate(v_points[:-1]):
            vp = v_points[i + 1]
            if vp - v < 1e-2:
                num = 501
            else:
                num = 201
            _v_vals = np.linspace(v, vp, num, endpoint=(i < len(v_points) - 2))
            _de_vals = np.array([de_v(v) for v in tqdm(_v_vals)])
            v_vals.append(_v_vals)
            de_vals.append(_de_vals)

        v_vals = np.concatenate(v_vals)
        de_vals = np.concatenate(de_vals)

        np.savez(data_filename, v=v_vals, de=de_vals)

    p = (1 - 4 * de_vals ** 2) / 2
    plt.plot(v_vals, p)
    plt.xlabel(r'$\Delta$')
    plt.ylabel(r'$\bar{P}$')  # (1-4(\frac{\partial \epsilon}{\partial \Delta})^2)/2$
    # plt.show()

    plt.tight_layout()
    plt.savefig('figures/Shirley/shirley_1965_fig2.pdf', dpi=400)


if __name__ == '__main__':
    shirley_1965_fig1()
    plt.show()
    shirley_1965_fig2()
    plt.show()