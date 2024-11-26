from scipy.constants import c, epsilon_0, e, m_e, N_A  # Import constants from scipy.constants
import quantities as pq  # Import quantities for handling units

from math import pi, log
# Overwrite the constants with units
c = c * pq.m / pq.s                 # Speed of light, unit: meters per second (m/s)
epsilon_0 = epsilon_0 * pq.F / pq.m # Vacuum permittivity, unit: farads per meter (F/m)
e = e * pq.C                        # Elementary charge, unit: coulombs (C)
m_e = m_e * pq.kg                   # Electron mass, unit: kilograms (kg)
N_A = N_A * pq.mol**-1              # Avogadro's number, unit: per mole (1/mol)

# Print the constants with units
print("Speed of light:", c)
print("Vacuum permittivity:", epsilon_0)
print("Elementary charge:", e)
print("Electron mass:", m_e)
print("Avogadro's number:", N_A)


# for the chlorophyll molecule
epsilon_max = 7e4 * pq.L / pq.mol / pq.cm
g_max = 660 * pq.nm / c
gamma = 20 * pq.nm / c

# oscillator strength
f = 2 * m_e * c * epsilon_0 * epsilon_max / (pi * N_A * e**2 * g_max)
f = f.simplified
print("Oscillator strength:", f)
