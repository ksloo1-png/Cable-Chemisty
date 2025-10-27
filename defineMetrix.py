import numpy as np
import matplotlib.pyplot as plt

def M_matrix(Zn, tau, omega):
    """
    Single-section transfer matrix (Eq. 5-style) with θ = ωτ:
        M = [[cosθ,  i Z sinθ],
             [ i(1/Z) sinθ,  cosθ]]
    """
    theta = omega * tau
    cos = np.cos(theta)
    sin = np.sin(theta)
    return np.array([
        [cos, 1j * Zn * sin],
        [1j * (1.0 / Zn) * sin, cos]
    ], dtype=complex)

def M_total(n, Z_list, tau, omega):
            
    if len(Z_list) < n:
        raise ValueError("Z_list must contain at least n elements.")

    M_tot = np.eye(2, dtype=complex)
    for i in range(n):
        Z = Z_list[i]
        M_tot = M_matrix(Z, tau, omega) @ M_tot
    return M_tot

def get_ABCD(n, Z_list, tau, omega):
    """
    Extract A, B, C, D parameters from the transfer matrix M.
    M is expected to be a 2x2 numpy array.
    """
    M_tot = M_total(n, Z_list, tau, omega)
    A = M_tot[0, 0]
    B = M_tot[0, 1]
    C = M_tot[1, 0]
    D = M_tot[1, 1]
    return A, B, C, D

def T_mag_S21(A, B, C, D, Z_in, Z_out):
    """
    Power-normalized transmission magnitude (S21):
        |T| = | 2*Z_in / (A*Z_in + B + C*Z_in*Z_out + D*Z_out) |
    This stays in [0,1] for passive, lossless networks.
    """
    num = 2.0 * Z_in
    den = A*Z_in + B + C*Z_in*Z_out + D*Z_out
    return abs(num / den)

def compute_T_mag(n, Z_list, tau, omega, Z_in=None, Z_out=None):
    """
    Convenience wrapper: build ABCD from (n, Z_list, tau, omega) and return |T|.
    If Z_in/Z_out not given, use the first/last Z in the list.
    """
    if Z_in is None:
        Z_in = Z_list[0]
    if Z_out is None:
        Z_out = Z_list[n-1]
    A, B, C, D = get_ABCD(n, Z_list, tau, omega)
    return T_mag_S21(A, B, C, D, Z_in, Z_out)

def random_Z_list(n, Z1=50.0, Z2=93.0, seed=None):
    rng = np.random.default_rng(seed)
    values = np.array([Z1, Z2], dtype=float)
    idx = rng.integers(0, 2, size=n)
    return list(values[idx])

if __name__ == "__main__":
    # Tau from tau * omega0 = pi/2, with f0 = 114 MHz
    f0 = 114e6
    omega0 = 2 * np.pi * f0
    tau = np.pi / (2.0 * omega0)

    # Build a random Z sequence
    n = 60                         # number of sections (edit as you like)
    Z_list = random_Z_list(n, 50.0, 93.0, seed=42)
    Z_in, Z_out = Z_list[0], Z_list[-1]

    # Frequency sweep: 0 → 300 MHz
    f = np.linspace(0.0, 300e6, 40001)   # many points for smooth curve
    omega = 2 * np.pi * f

    # Compute |T|(f)
    Tmag = np.empty_like(f, dtype=float)
    for i, w in enumerate(omega):
        Tmag[i] = compute_T_mag(n, Z_list, tau, w, Z_in=Z_in, Z_out=Z_out)

    # Plot
    plt.figure(figsize=(9, 5))
    plt.plot(f/1e6, Tmag, linewidth=1.1)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel(r"$|T|$")
    plt.title(rf"|T| vs Frequency (n={n}, $f_0$=114 MHz, $\tau=\pi/(2\omega_0)$)")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", linewidth=0.6)
    plt.tight_layout()
    plt.show()
