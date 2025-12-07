from sauter_pulse import sauter_pulse, load_double_sauter
from scipy.constants import m_e, eV, c

m_eV = m_e * c ** 2 / eV

# # The following parameters show enhancement, yet phase not changed much
# tau1 = 40
# tau2 = tau1 * 0.01
# t0_tau1 = 0
# t0 = t0_tau1 * tau1  # shift
# e1 = 0.5  # field strength, in unit of E_c
# e2 = 1 * 0.05


# tau1 = 12
# tau2 = tau1 * 0.01
# t0_tau1 = 0
# t0 = t0_tau1 * tau1  # shift
# e1 = 0.25  # field strength, in unit of E_c
# e2 = 1 * 0.025


# tau1 = 20
# tau2 = tau1 * 0.1
# t0_tau1 = 0
# t0 = t0_tau1 * tau1  # shift
# e1 = 0.5  # field strength, in unit of E_c
# e2 = 1 * 0.05


# tau1 = 0.2
# tau2 = tau1 * 0.1
# t0_tau1 = 0
# t0 = t0_tau1 * tau1  # shift
# e1 = 0.5  # field strength, in unit of E_c
# e2 = 0 * 0.05

# Parameters used in numerics verification
e1 = 0.25
tau1_eV = 1e-4  # in unit of eV
tau1 = tau1_eV * m_eV  # in unit of 1/m
e2 = 1 * 0.025
tau2_eV = 2e-6  # in unit of eV
tau2 = tau2_eV * m_eV  # in unit of 1/m
t0_tau1 = 0  # -0.25
t0 = t0_tau1 * tau1  # shift


params = {
    'tau1': tau1,
    'e1': e1,
    'tau2': tau2,
    'e2': e2,
    't0': t0_tau1 * tau1
}

A1, E1 = sauter_pulse(e1, tau1)
A2, E2 = sauter_pulse(e2, tau2)


def A_func(t): return A1(t) + A2(t - t0)


def E_func(t): return E1(t) + E2(t - t0)
