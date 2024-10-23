import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def draw_heatmap(ts_pn, ts=None, ax=None):
    """
    Visualize Pn(t) data in a heatmap.
    """
    if ax is None:
        _, ax = plt.subplots()

    nt, nn = ts_pn.shape
    if ts is None:
        ts = np.arange(nt)

    # Draw a heatmap with the numeric values in each cell
    sns.heatmap(ts_pn, cmap='Blues', annot=False, cbar_kws={"shrink": 0.8}, ax=ax)

    # remove the axis
    # ax.set_axis_off()

    plt.xlabel('n')
    plt.ylabel('t')

    # Set the axis ticks, only show up to 10 ticks
    Nt, Nn = min(10, nt), min(10, nn)
    t_spacing = nt // Nt
    n_spacing = nn // Nn
    t_ticks = list(range(0, nt, t_spacing))

    ax.set_xticks(ticks=[i + 0.5 for i in range(0, nn, n_spacing)],
                  labels=[str(i + 1) for i in range(0, nn, n_spacing)])
    ax.set_yticks(ticks=t_ticks,
                  labels=[f'{ts[t]:.2f}' for t in t_ticks])

    ax.set_title('Time evolution of probability distribution $P_n(t)$')
    return ax


def test():
    fig, ax = plt.subplots()

    # 生成一些随机数据
    data = np.random.rand(100, 20)
    draw_heatmap(data, ax=ax)
    plt.savefig('time_evolution.png', dpi=400)


if __name__ == '__main__':
    test()
