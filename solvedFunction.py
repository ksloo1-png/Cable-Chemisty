import numpy as np
import matplotlib.pyplot as plt
from defineMetrix import M_matrix

# --- Random Z helper ---
def random_Z_list(length, values=(50.0, 93.0), seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
        return list(rng.choice(values, size=length))
    else:
        rng = np.random.default_rng()
        return list(rng.choice(values, size=length))

# --- ABCD for a SINGLE omega ---
def get_transfer_ABCD(tau, Z_list, omega_scalar):
    M = np.eye(2, dtype=complex)
    for Z in Z_list:
        # If your M_matrix signature is (Z, d, c, omega), change the next line accordingly.
        M = M_matrix(Z, tau, omega_scalar) @ M
    A, B, C, D = M[0,0], M[0,1], M[1,0], M[1,1]
    return A, B, C, D

def transmission_magnitude(A, B, C, D, Z_in, Z_out):
    # |T| = |(A*D*Z_in - B*C*Z_out) / (D*Z_out - C*Z_in*Z_out - B + A*Z_in)|
    numerator = A*D*Z_in - B*C*Z_out
    denominator = D*Z_out - C*Z_in*Z_out - B + A*Z_in
    return abs(numerator / denominator)

def transmission_spectrum(tau, Z_list, omega_array):
    Zin, Zout = Z_list[0], Z_list[-1]
    T = np.empty_like(omega_array, dtype=float)
    for i, w in enumerate(omega_array):
        A, B, C, D = get_transfer_ABCD(tau, Z_list, w)  # <-- w is scalar
        T[i] = transmission_magnitude(A, B, C, D, Zin, Zout)
    return T

if __name__ == "__main__":
    # Build random Z sequence (edit length/seed as you wish)
    Z_list = random_Z_list(length=10, values=(50.0, 93.0), seed=42)

    d = 0.5            # m
    c = 3e8            # m/s  (note: changed from 2e8 to 3e8)
    tau = d / c

    # Frequency sweep
    f = np.linspace(0, 600e6, 1000)   # Hz
    omega = 2*np.pi*f

    # Compute |T|
    Tmag = transmission_spectrum(tau, Z_list, omega)

    # Plot
    plt.figure(figsize=(9,5))
    plt.plot(omega/1e6, Tmag)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel(r"$|T|$")
    plt.title(r"Transmission magnitude $|T|$ vs Frequency")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.show()
