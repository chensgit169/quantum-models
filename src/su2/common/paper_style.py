import matplotlib as mpl


def set_paper_style():
    mpl.rcParams.update({
        # "font.family": "serif",
        "font.size": 16,
        "axes.labelsize": 21,
        # "legend.fontsize": 16,
        "xtick.labelsize": 21,
        "ytick.labelsize": 21,
        "lines.linewidth": 2,
        "lines.markersize": 6,
        "axes.linewidth": 2,
        "figure.figsize": (3.3, 2.2),  # 单栏尺寸
        "figure.dpi": 400,
    })


if __name__ == '__main__':
    set_paper_style()
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.plot(x, y, label='sin(x)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Example Plot with Paper Style')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('example_paper_style.pdf', dpi=300)
    plt.show()