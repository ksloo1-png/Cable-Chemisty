import numpy as np
import matplotlib.pyplot as plt

# Given epsilon values
epsilon = np.array([0.000902, 0.000778, 0.000603, 0.000428, 0.000317])

# Build symmetric data
eps_all = np.concatenate((-epsilon[::-1], epsilon))
energies = eps_all  # y-values (±ε)
x = eps_all         # x-axis values (ε itself)

# Plot
plt.figure(figsize=(6,4))
plt.stem(x, energies, linefmt='b-', markerfmt='bo', basefmt=' ')
plt.axhline(0, color='gray', lw=1)
plt.xlabel("ε (Energy eigenvalue)")
plt.ylabel("Energy level")
plt.title("±ε spectrum (ε on x-axis)")
plt.grid(True, alpha=0.4)
plt.show()
