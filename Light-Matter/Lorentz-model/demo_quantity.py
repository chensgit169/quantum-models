from scipy.constants import c, epsilon_0, e, m_e, N_A
import quantities as pq

from quantities.constants import c, elementary_charge, electron_mass, vacuum_permittivity, avogadro_number

# constants with units
# Speed of light in vacuum
c = c * pq.meter / pq.second  # m/s
epsilon_0 = epsilon_0 * pq.farad / pq.meter  # F/m


def test():
    # Print the constants and their units
    print(f"Speed of light (c): {c:.2e} m/s")
    print(f"Vacuum permittivity (epsilon_0): {epsilon_0:.2e} F/m")
    print(f"Electron charge (e): {e:.2e} C")
    print(f"Electron mass (m_e): {m_e:.2e} kg")

    # Conversion factors
    m_to_cm = 100  # 1 m = 100 cm
    kg_to_g = 1000  # 1 kg = 1000 g
    C_to_esu = 3.33564e9  # 1 C = 3.33564e9 esu

    # Convert the constants to CGS units
    c_cgs = c * m_to_cm / pq.s  # Speed of light in cm/s
    epsilon_0_cgs = epsilon_0 * (C_to_esu ** 2) / (m_to_cm ** 3)  # Vacuum permittivity in esu^2 / cm^3
    e_cgs = e * C_to_esu  # Elementary charge in esu
    m_e_cgs = m_e * kg_to_g  # Electron mass in g

    # Output the constants in CGS units
    print(f"Speed of light (c): {c_cgs:.2e} cm/s")
    print(f"Vacuum permittivity (epsilon_0): {epsilon_0_cgs:.2e} esu² / cm³")
    print(f"Electron charge (e): {e_cgs:.2e} esu")
    print(f"Avogadro's number (N_A): {N_A:.2e} 1/mol")
    print(f"Electron mass (m_e): {m_e_cgs:.2e} g")


# 定义物理量
distance = 10 * pq.meter  # 10米
time = 5 * pq.second  # 5秒

# 计算速度
speed = distance / time  # 速度单位会自动为 m/s
print(speed)  # 输出: 2.0 m/s

# 单位转换
distance_km = distance.rescale(pq.kilometer)  # 转换为公里
print(distance_km)  # 输出: 0.01 km

# 单位校验
mass = 70 * pq.kg  # 70千克
# volume = mass / distance       # 这会抛出异常，因为单位不匹配
