from su2.common.su2_integrator import evolve
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

from v_a import h_x, x_t, _gamma, lz_p, wkb_p


mpl.rcParams.update({
    # "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 21,
    "legend.fontsize": 21,
    "xtick.labelsize": 21,
    "ytick.labelsize": 21,
    "lines.linewidth": 3,
    "lines.markersize": 6,
    "axes.linewidth": 2,
    "figure.figsize": (8, 6),  # 单栏尺寸
    "figure.dpi": 400,
})


def demo(d=1, v=1):
    g = _gamma(v, d)
    p_exact = lz_p(g)
    print(f'v={v}, g={g:.3f}, p_exact={p_exact:.3e}')

    t_lim = 1000
    ts = np.linspace(-t_lim, t_lim, 50000)
    xs = x_t(ts, v, d)
    mask = np.abs(ts) < 10
    ts_demo = ts[mask]

    p_1st, p_wkb = wkb_p(xs, g)
    p_1st = p_1st[mask]
    p_wkb = p_wkb[mask]

    psi_0 = np.array([1, 0])
    psi_t = evolve(psi_0, xs[-1], xs[0], len(xs), lambda x: h_x(x, g))
    beta_t = psi_t[:, 1][mask]
    p_num = np.abs(beta_t) ** 2

    plt.axhline(p_exact, color='grey', linestyle='-.', label=r'$\exp(-2\pi \gamma)$')
    plt.plot(ts_demo, p_num, label=r'Exact')
    plt.plot(ts_demo, p_1st, '--', label=r'FMA')
    plt.plot(ts_demo, p_wkb, ':', label=r'WKB')
    plt.xlabel('t')
    plt.ylabel(r'$P(t)=|\beta(t)|^2$')
    plt.xlim(min(ts_demo), max(ts_demo))
    # plt.title('Landau-Zener Transition Amplitude')

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figures/lz_evolution_d={d:.2f}_v={v:.2f}.pdf', dpi=400)

    plt.show()


if __name__ == '__main__':
    for v in [0.25, 0.5, 1]:
        demo(v=v)
