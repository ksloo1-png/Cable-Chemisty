import numpy as np
import matplotlib.pyplot as plt

# ---------- Single-section transfer matrix ----------
def M_matrix(Zn, tau, omega):
    """
    Single-section transfer matrix with θ = ωτ:
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

# ---------- Cascade ----------
def M_total(n, Z_list, tau, omega):
    """Compute total transfer matrix for n sections."""
    if len(Z_list) < n:
        raise ValueError("Z_list must contain at least n elements.")
    M_tot = np.eye(2, dtype=complex)
    for i in range(n):
        M_tot = M_matrix(Z_list[i], tau, omega) @ M_tot
    return M_tot

def get_ABCD(n, Z_list, tau, omega):
    M_tot = M_total(n, Z_list, tau, omega)
    A, B = M_tot[0, 0], M_tot[0, 1]
    C, D = M_tot[1, 0], M_tot[1, 1]
    return A, B, C, D

# ---------- Transmission coefficient ----------
def T_mag_S21(A, B, C, D, Z_in, Z_out):
    """Power-normalized |T| (S21)"""
    num = 2.0 * Z_in
    den = A*Z_in + B + C*Z_in*Z_out + D*Z_out
    return abs(num / den)

def compute_T_mag(n, Z_list, tau, omega, Z_in=None, Z_out=None):
    if Z_in is None:
        Z_in = Z_list[0]
    if Z_out is None:
        Z_out = Z_list[-1]
    A, B, C, D = get_ABCD(n, Z_list, tau, omega)
    return T_mag_S21(A, B, C, D, Z_in, Z_out)

# ---------- Z list generators ----------
def random_Z_list(n, Z1=50.0, Z2=93.0, seed=None):
    rng = np.random.default_rng(seed)
    values = np.array([Z1, Z2], dtype=float)
    idx = rng.integers(0, 2, size=n)
    return list(values[idx])

def repeating_Z_list(n, Z1=50.0, Z2=93.0):
    """Return alternating [Z1, Z2, Z1, Z2, ...] up to n elements."""
    pattern = [Z1, Z2]
    return [pattern[i % 2] for i in range(n)]

# ---------- Main plotting function ----------
if __name__ == "__main__":
    # Set parameters
    f0 = 114e6
    omega0 = 2 * np.pi * f0
    tau = np.pi / (2.0 * omega0)   # quarter-wave condition

    n = 100          # number of sections
    case = "random"  # choose: "random" or "repeat"

    if case == "random":
        Z_list = random_Z_list(n, 50.0, 93.0, seed=42)
    elif case == "repeat":
        Z_list = repeating_Z_list(n, 50.0, 93.0)
    else:
        raise ValueError("case must be 'random' or 'repeat'.")

    #print(f"Z_list (first 12 shown): {Z_list[:12]} ...")

    Z_in, Z_out = Z_list[0], Z_list[-1]

    # Frequency sweep: 0 → 300 MHz
    f = np.linspace(0.0, 300e6, 40001)
    omega = 2 * np.pi * f

    # Compute |T|(f)
    Tmag = np.empty_like(f, dtype=float)
    for i, w in enumerate(omega):
        Tmag[i] = compute_T_mag(n, Z_list, tau, w, Z_in=Z_in, Z_out=Z_out)

    # ---------- Plot ----------
    plt.figure(figsize=(9, 5))
    plt.plot(f / 1e6, Tmag, linewidth=1.1,
             label=f"{case.title()} Z pattern")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel(r"$|T|$")
    plt.title(rf"|T| vs Frequency (n={n})")
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle="--", linewidth=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()
