import numpy as np

def M_matrix(Zn, tau, omega):
    """Return the 2x2 transfer matrix M_n for one uniform section."""
    theta = omega * tau
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([
        [cos, 1j * Zn * sin],
        [-(1j / Zn) * sin, cos]
    ])

# Example parameters
omega = 2 * np.pi * 1.14e7    # 1.14 MHz angular frequency
d = 0.5                    # section length (m)
c = 2e8                    # wave speed (m/s)
tao = d / c                 # time delay

# Two impedances
Z1 = 50.0
Z2 = 93.0

# Compute the individual M matrices
M1 = M_matrix(Z1, tao, omega)
M2 = M_matrix(Z2, tao, omega)

# Multiply to get total transfer matrix (M_total = M2 * M1)
#M_total = M2 @ M1

# Display results
np.set_printoptions(precision=4, suppress=True)
#print("M1 =\n", M1)
#print("\nM2 =\n", M2)

"""
M_total = np.eye(2, dtype=complex)
for Zn in [50, 93, 50, 93]:
    M_total = M_matrix(Zn, tao, omega) @ M_total
print("\nM_total =\n", M_total)
"""