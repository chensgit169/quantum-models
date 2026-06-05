import numpy as np
import matplotlib.pyplot as plt


def g(x):
    return np.arccos(x) / np.sqrt(1 - x ** 2)


# main domain avoiding endpoints
x_main = np.linspace(-0.9999, 0.9999, 2000)
y_main = g(x_main)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(x_main, y_main, lw=1)


plt.tight_layout()
plt.show()
