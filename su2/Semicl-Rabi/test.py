import numpy as np
import matplotlib.pyplot as plt

xs = np.linspace(-1, 2, 100)

ys = np.arccos(xs.astype(complex))

plt.plot(xs, ys.real, label='Re[arccos(x)]')
plt.plot(xs, ys.imag, label='Im[arccos(x)]')
plt.xlabel('x')
plt.legend()
plt.grid(True)
plt.show()