import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def draw_heatmap(ts_pn, ax=None):
    """
    Visualize Pn(t) data in a heatmap.
    """
    if ax is None:
        _, ax = plt.subplots()

    nt, nn = ts_pn.shape

    # Draw a heatmap with the numeric values in each cell
    sns.heatmap(ts_pn, cmap='Blues', annot=False, cbar_kws={"shrink": 0.8}, ax=ax)

    # remove the axis
    # ax.set_axis_off()

    plt.xlabel('n')
    plt.ylabel('t')

    ax.set_xticks(ticks=[i + 0.5 for i in range(nn)],
                  labels=[str(i+1) for i in range(nn)])

    ax.set_title('Time evolution of probability distribution $P_n(t)$')
    return ax


def test():
    fig, ax = plt.subplots()

    # 生成一些随机数据
    data = np.random.rand(100, 20)
    draw_heatmap(data, ax)
    plt.savefig('time_evolution.png', dpi=400)


if __name__ == '__main__':
    test()