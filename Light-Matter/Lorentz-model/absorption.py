from math import pi, log
from scipy.constants import N_A, c, hbar

from quantities import Quantity

hbar_SI = Quantity(hbar, "J·s")
c_SI = Quantity(c, "m/s")
clearclear
hbar_c = hbar_SI * c_SI

# Convert hbar * c to erg·cm (CGS units)
hbar_c = hbar_c.rescale("erg*cm")


coeff = hbar_c.magnitude / (2 * pi * N_A)
print(coeff)